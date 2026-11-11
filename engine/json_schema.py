"""Compile a JSON Schema into a pushdown grammar (Phase 2).

Maps schema constructs onto grammar rules: objects to ``{ key : value , ... }``
with required keys in declared order and optional keys that may be skipped
(with the comma logic handled so output is always well-formed); arrays to
``[ value , ... ]`` of the item type; and scalars to shared STRING / INT /
NUMBER / boolean / null / enum matchers. Insignificant JSON whitespace is
allowed between tokens, so the model may emit ``{"a": 1}`` or ``{ "a":1 }``.

Nested objects and arrays are emitted as ``Call``\\ s into their own rules, so
real nesting runs through the PDA stack. ``build_json_processor`` returns a
ready LogitsProcessor; generation constrained by it parses as the schema by
construction.

Scope: fixed schemas with object/array/string/integer/number/boolean/null,
``enum`` and ``const``. Recursive ``$ref`` is out of scope (Phase 3 EBNF covers
recursion); ``additionalProperties`` is treated as closed (only listed keys).
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from engine.pushdown import (
    Alt,
    Call,
    Chars,
    Empty,
    Grammar,
    Lit,
    Opt,
    Plus,
    Seq,
    Star,
    any_but,
    char_set,
    compile_grammar,
    lit_char,
)

# Insignificant whitespace (reusable node; compile mints fresh states per use).
_WS = Star(char_set(" \t\n\r"))
_DIGIT = char_set("0123456789")
_NONZERO = char_set("123456789")
_HEX = char_set("0123456789abcdefABCDEF")


def _scalar_rules() -> Dict[str, object]:
    """The shared leaf rules every schema reuses."""
    string = Seq([
        Lit('"'),
        Star(Alt([
            any_but({'"', "\\"}),  # ordinary character (a '{' here is just text)
            Seq([Lit("\\"), Alt([
                char_set('"\\/bfnrt'),
                Seq([Lit("u"), _HEX, _HEX, _HEX, _HEX]),
            ])]),
        ])),
        Lit('"'),
    ])
    int_core = Seq([
        Opt(Lit("-")),
        Alt([Lit("0"), Seq([_NONZERO, Star(_DIGIT)])]),
    ])
    number = Seq([
        int_core,
        Opt(Seq([Lit("."), Plus(_DIGIT)])),
        Opt(Seq([char_set("eE"), Opt(char_set("+-")), Plus(_DIGIT)])),
    ])
    return {"STRING": string, "INT": int_core, "NUMBER": number}


class _SchemaCompiler:
    def __init__(self):
        self.rules: Dict[str, object] = _scalar_rules()
        self._counter = 0

    def _fresh(self, hint: str) -> str:
        self._counter += 1
        return f"{hint}{self._counter}"

    # -- value of any schema -------------------------------------------------
    def value(self, schema: dict) -> object:
        if schema is None:
            schema = {}
        if "const" in schema:
            return Lit(json.dumps(schema["const"]))
        if "enum" in schema:
            return Alt([Lit(json.dumps(v)) for v in schema["enum"]])

        t = schema.get("type")
        if isinstance(t, list):
            return Alt([self.value({**schema, "type": one}) for one in t])

        if t == "object":
            return Call(self._object_rule(schema))
        if t == "array":
            return Call(self._array_rule(schema))
        if t == "string":
            return Call("STRING")
        if t == "integer":
            return Call("INT")
        if t == "number":
            return Call("NUMBER")
        if t == "boolean":
            return Alt([Lit("true"), Lit("false")])
        if t == "null":
            return Lit("null")
        # No type/enum/const: permit any scalar JSON value (no nested container).
        return Alt([
            Call("STRING"), Call("NUMBER"),
            Lit("true"), Lit("false"), Lit("null"),
        ])

    # -- object --------------------------------------------------------------
    def _object_rule(self, schema: dict) -> str:
        name = self._fresh("obj")
        props: dict = schema.get("properties", {})
        required = set(schema.get("required", []))
        keys: List[Tuple[str, dict, bool]] = [
            (k, sub, k in required) for k, sub in props.items()
        ]
        body = Seq([Lit("{"), _WS, self._members(keys), _WS, Lit("}")])
        self.rules[name] = body
        return name

    def _member(self, key: str, sub: dict) -> object:
        return Seq([Lit(json.dumps(key)), _WS, Lit(":"), _WS, self.value(sub)])

    def _members(self, keys: List[Tuple[str, dict, bool]], emitted: bool = False) -> object:
        """Emit keys in order; a comma precedes every key after the first
        emitted one. Optional keys branch into present/absent, so the comma
        placement is always correct regardless of which optionals appear."""
        if not keys:
            return Empty()
        (key, sub, required), rest = keys[0], keys[1:]
        member = self._member(key, sub)
        with_sep = member if not emitted else Seq([_WS, Lit(","), _WS, member])
        present = Seq([with_sep, self._members(rest, True)])
        if required:
            return present
        absent = self._members(rest, emitted)
        return Alt([present, absent])

    # -- array ---------------------------------------------------------------
    def _array_rule(self, schema: dict) -> str:
        name = self._fresh("arr")
        items = schema.get("items", {})
        element = self.value(items)
        more = Star(Seq([_WS, Lit(","), _WS, self.value(items)]))
        body = Seq([
            Lit("["), _WS,
            Opt(Seq([element, more])),
            _WS, Lit("]"),
        ])
        self.rules[name] = body
        return name


def schema_to_grammar(schema: dict) -> Grammar:
    c = _SchemaCompiler()
    c.rules["MAIN"] = Seq([_WS, c.value(schema), _WS])
    return Grammar(rules=c.rules, start="MAIN")


def build_json_automaton(schema: dict):
    """Compile a schema straight to a :class:`~engine.pushdown.PDA`."""
    return compile_grammar(schema_to_grammar(schema))


def build_json_processor(schema: dict, vocab, indexed: bool = False, precompute: bool = False):
    """Return a LogitsProcessor that constrains generation to ``schema``.

    ``indexed`` adds the Phase-3 per-state token cache (:class:`TokenIndex`),
    which removes the per-step full-vocab scan; ``precompute`` fills it eagerly
    by walking the reachable state graph once up front."""
    from engine.masker import AutomatonLogitsProcessor
    from engine.token_bridge import allowed_token_ids

    automaton = build_json_automaton(schema)
    allowed_fn = allowed_token_ids
    if indexed:
        from engine.token_index import TokenIndex

        idx = TokenIndex(automaton, vocab)
        if precompute:
            idx.precompute()
        allowed_fn = idx
    return AutomatonLogitsProcessor(automaton, vocab, allowed_fn=allowed_fn)
