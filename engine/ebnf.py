"""EBNF grammar frontend (Phase 3) — a context-free grammar compiled to the
pushdown automaton, for constraining output to a tool-call DSL.

This is where the PDA's stack genuinely earns its keep: a grammar rule may
reference itself (directly or through others), so nesting depth is unbounded and
no finite DFA can represent it. For example a ``value`` that may be a ``call``
makes ``add(mul(2,3), neg(4))`` legal at any depth — the stack tracks it.

Supported EBNF subset
    rule = expression ;            rule definition (first rule is the start)
    a | b                          alternation
    a b   (or  a , b)              concatenation
    "lit" / 'lit'                  terminal string
    /regex/                        terminal pattern (reuses the Phase-1 parser)
    ( ... )                        grouping
    [ ... ]                        optional   { ... }  repetition (zero+)
    x?  x*  x+                     postfix optional / star / plus
    (* comment *)                  comments
    IDENT                          reference to another rule

Everything compiles to the combinators in :mod:`engine.pushdown`, so the token
bridge, masker, and token index all apply unchanged.
"""

from __future__ import annotations

import re
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
    compile_grammar,
)
from engine.regex_compiler import parse_regex_ast
import engine.regex_compiler as _rx


# --------------------------------------------------------------------------- #
# /regex/ terminal -> pushdown combinators (reuse the Phase-1 regex AST)
# --------------------------------------------------------------------------- #
def _regex_node_to_combinator(node):
    if isinstance(node, _rx._Empty):
        return Empty()
    if isinstance(node, _rx._Leaf):
        spec = node.spec
        if spec.kind == "lit":
            return Chars(("lit", frozenset(spec.chars)))
        if spec.kind == "any":
            return Chars(("any",))
        # class
        if spec.negated:
            return Chars(("anybut", frozenset(spec.chars)))
        return Chars(("set", frozenset(spec.chars)))
    if isinstance(node, _rx._Concat):
        return Seq([_regex_node_to_combinator(p) for p in node.parts])
    if isinstance(node, _rx._Alt):
        return Alt([_regex_node_to_combinator(o) for o in node.options])
    if isinstance(node, _rx._Repeat):
        inner = _regex_node_to_combinator(node.node)
        m, n = node.m, node.n
        if m == 0 and n is None:
            return Star(inner)
        if m == 1 and n is None:
            return Plus(inner)
        if m == 0 and n == 1:
            return Opt(inner)
        if n is None:
            return Seq([inner] * m + [Star(inner)])
        return Seq([inner] * m + [Opt(inner)] * (n - m))
    raise TypeError(f"unhandled regex node {node!r}")


def regex_terminal(pattern: str):
    ast, _alphabet = parse_regex_ast(pattern)
    return _regex_node_to_combinator(ast)


# --------------------------------------------------------------------------- #
# EBNF tokenizer
# --------------------------------------------------------------------------- #
_TOKEN = re.compile(
    r"""
      (?P<ws>\s+)
    | (?P<comment>\(\*.*?\*\))
    | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
    | (?P<dstring>"(?:\\.|[^"\\])*")
    | (?P<sstring>'(?:\\.|[^'\\])*')
    | (?P<regex>/(?:\\.|[^/\\])+/)
    | (?P<punct>[=;|()\[\]{}*+?,])
    """,
    re.VERBOSE | re.DOTALL,
)

_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "'": "'", "\\": "\\", "/": "/"}


def _unescape(s: str) -> str:
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            out.append(_ESCAPES.get(s[i + 1], s[i + 1]))
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _tokenize(text: str) -> List[Tuple[str, str]]:
    tokens: List[Tuple[str, str]] = []
    pos = 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m:
            raise ValueError(f"EBNF: unexpected character {text[pos]!r} at {pos}")
        pos = m.end()
        kind = m.lastgroup
        if kind in ("ws", "comment"):
            continue
        val = m.group()
        if kind in ("dstring", "sstring"):
            tokens.append(("string", _unescape(val[1:-1])))
        elif kind == "regex":
            tokens.append(("regex", val[1:-1]))
        elif kind == "ident":
            tokens.append(("ident", val))
        else:  # punct
            tokens.append((val, val))
    tokens.append(("eof", ""))
    return tokens


# --------------------------------------------------------------------------- #
# EBNF parser -> Grammar
# --------------------------------------------------------------------------- #
class _EBNFParser:
    def __init__(self, tokens: List[Tuple[str, str]]):
        self.toks = tokens
        self.i = 0

    def _peek(self) -> Tuple[str, str]:
        return self.toks[self.i]

    def _next(self) -> Tuple[str, str]:
        t = self.toks[self.i]
        self.i += 1
        return t

    def _expect(self, kind: str) -> Tuple[str, str]:
        t = self._next()
        if t[0] != kind:
            raise ValueError(f"EBNF: expected {kind!r}, got {t[0]!r} ({t[1]!r})")
        return t

    def parse(self) -> Grammar:
        rules: Dict[str, object] = {}
        start: Optional[str] = None
        while self._peek()[0] != "eof":
            name = self._expect("ident")[1]
            self._expect("=")
            body = self._alt()
            self._expect(";")
            rules[name] = body
            if start is None:
                start = name
        if start is None:
            raise ValueError("EBNF: no rules defined")
        missing = {
            r for body in rules.values() for r in _referenced(body) if r not in rules
        }
        if missing:
            raise ValueError(f"EBNF: undefined rule(s): {sorted(missing)}")
        return Grammar(rules=rules, start=start)

    def _alt(self):
        opts = [self._seq()]
        while self._peek()[0] == "|":
            self._next()
            opts.append(self._seq())
        return opts[0] if len(opts) == 1 else Alt(opts)

    _STOP = {"|", ")", "]", "}", ";", "eof"}

    def _seq(self):
        parts = []
        while self._peek()[0] not in self._STOP:
            if self._peek()[0] == ",":  # optional concatenation separator
                self._next()
                continue
            parts.append(self._postfix())
        if not parts:
            return Empty()
        return parts[0] if len(parts) == 1 else Seq(parts)

    def _postfix(self):
        node = self._atom()
        while self._peek()[0] in ("*", "+", "?"):
            op = self._next()[0]
            node = {"*": Star, "+": Plus, "?": Opt}[op](node)
        return node

    def _atom(self):
        kind, val = self._peek()
        if kind == "(":
            self._next()
            node = self._alt()
            self._expect(")")
            return node
        if kind == "[":
            self._next()
            node = self._alt()
            self._expect("]")
            return Opt(node)
        if kind == "{":
            self._next()
            node = self._alt()
            self._expect("}")
            return Star(node)
        if kind == "string":
            self._next()
            return Lit(val)
        if kind == "regex":
            self._next()
            return regex_terminal(val)
        if kind == "ident":
            self._next()
            return Call(val)
        raise ValueError(f"EBNF: unexpected token {kind!r} ({val!r})")


def _referenced(node) -> set:
    """Collect rule names referenced by ``Call`` anywhere in a combinator tree."""
    out = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, Call):
            out.add(n.rule)
        elif isinstance(n, (Seq, Alt)):
            stack.extend(n.parts if isinstance(n, Seq) else n.options)
        elif isinstance(n, (Opt, Star, Plus)):
            stack.append(n.node)
    return out


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def parse_ebnf(text: str) -> Grammar:
    return _EBNFParser(_tokenize(text)).parse()


def compile_ebnf(text: str):
    """Compile EBNF source into a :class:`~engine.pushdown.PDA`."""
    return compile_grammar(parse_ebnf(text))


def load_ebnf(path: str):
    with open(path) as f:
        return compile_ebnf(f.read())


def build_ebnf_processor(text: str, vocab, indexed: bool = False, precompute: bool = False):
    """Return a LogitsProcessor constraining generation to an EBNF grammar."""
    from engine.masker import AutomatonLogitsProcessor
    from engine.token_bridge import allowed_token_ids

    automaton = compile_ebnf(text)
    allowed_fn = allowed_token_ids
    if indexed:
        from engine.token_index import TokenIndex

        idx = TokenIndex(automaton, vocab)
        if precompute:
            idx.precompute()
        allowed_fn = idx
    return AutomatonLogitsProcessor(automaton, vocab, allowed_fn=allowed_fn)
