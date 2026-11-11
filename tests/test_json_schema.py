"""JSON-schema pushdown grammar (Phase 2): acceptance, edge cases, and the
end-to-end masking invariant, all without a model."""

import json
import random

import jsonschema
import pytest
import torch

from engine.json_schema import build_json_automaton
from engine.token_bridge import TokenVocab
from engine.masker import AutomatonLogitsProcessor


FLAT = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
    "additionalProperties": False,
}
ENUM = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["active", "inactive", "pending"]},
        "id": {"type": "integer"},
    },
    "required": ["status", "id"],
    "additionalProperties": False,
}
ARRAY = {
    "type": "object",
    "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
    "required": ["tags"],
    "additionalProperties": False,
}
OPTIONAL = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "subtitle": {"type": "string"}},
    "required": ["title"],
    "additionalProperties": False,
}
NESTED = {
    "type": "object",
    "properties": {
        "user": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "active": {"type": "boolean"}},
            "required": ["name", "active"],
            "additionalProperties": False,
        },
        "score": {"type": "number"},
    },
    "required": ["user", "score"],
    "additionalProperties": False,
}
EMPTY_OBJ = {"type": "object", "properties": {}, "additionalProperties": False}


@pytest.mark.parametrize(
    "schema,accepts,rejects",
    [
        (FLAT,
         ['{"name": "Ada", "age": 36}', '{"name":"Ada","age":36}', '{ "name" : "x" , "age" : -3 }'],
         ['{"name": "Ada"}', '{"age": 36, "name": "Ada"}', '{"name": "Ada", "age": "x"}', '{"name": "Ada", "age": 36']),
        (ENUM,
         ['{"status": "active", "id": 7}', '{"status":"pending","id":0}'],
         ['{"status": "paused", "id": 7}', '{"status": "active", "id": 7.5}']),
        (ARRAY,
         ['{"tags": []}', '{"tags": ["a"]}', '{"tags": ["a", "b", "c"]}'],
         ['{"tags": ["a" "b"]}', '{"tags": [1]}', '{"tags": ["a",]}']),
        (OPTIONAL,
         ['{"title": "Dune"}', '{"title": "Dune", "subtitle": "pt 1"}'],
         ['{"subtitle": "x"}', '{}']),
        (NESTED,
         ['{"user": {"name": "Grace", "active": true}, "score": 9.5}',
          '{"user": {"name": "a{b}c", "active": false}, "score": -3.2e10}'],
         ['{"user": {"name": "Grace"}, "score": 9.5}',
          '{"user": {"name": "x", "active": true} "score": 9.5}']),
        (EMPTY_OBJ, ["{}", "{   }"], ['{"a": 1}']),
    ],
)
def test_accept_reject(schema, accepts, rejects):
    pda = build_json_automaton(schema)
    for s in accepts:
        assert pda.accepts(s), f"should accept {s!r}"
        # Cross-check our acceptance against a real validator.
        jsonschema.validate(json.loads(s), schema)
    for s in rejects:
        assert not pda.accepts(s), f"should reject {s!r}"


def test_brace_inside_string_does_not_move_stack():
    # A '{' inside a string value must be ordinary text, not a nesting push.
    pda = build_json_automaton(FLAT)
    assert pda.accepts('{"name": "a{b}[c]", "age": 1}')


def test_escaped_quote_in_string():
    pda = build_json_automaton(FLAT)
    assert pda.accepts('{"name": "say \\"hi\\"", "age": 1}')
    jsonschema.validate(json.loads('{"name": "say \\"hi\\"", "age": 1}'), FLAT)


def test_nested_array_of_objects():
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"sku": {"type": "string"}, "qty": {"type": "integer"}},
                    "required": ["sku", "qty"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }
    pda = build_json_automaton(schema)
    assert pda.accepts('{"items": [{"sku": "a", "qty": 1}, {"sku": "b", "qty": 2}]}')
    assert pda.accepts('{"items": []}')
    assert not pda.accepts('{"items": [{"sku": "a"}]}')  # missing qty


def test_number_formats():
    schema = {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"],
              "additionalProperties": False}
    pda = build_json_automaton(schema)
    for v in ["0", "-1", "3.14", "-2.5e10", "1E5", "6.022e23"]:
        assert pda.accepts('{"x": %s}' % v), v
    for v in ["01", "1.", ".5", "1e", "--1", "+1"]:
        assert not pda.accepts('{"x": %s}' % v), v


# --------------------------------------------------------------------------- #
# End-to-end masking invariant (the real proof), still model-free.
# --------------------------------------------------------------------------- #
def _fake_vocab():
    singles = [chr(c) for c in range(32, 127)]
    multis = ["true", "false", "null", '"name"', '"age"', '"user"', '"active"',
              '"score"', '"tags"', "Grace", '{"', '":', '",', '"]', '"}']
    pieces = singles + multis
    return TokenVocab.from_pairs(list(enumerate(pieces)), eos_id=len(pieces)), pieces


def test_masking_never_deadlocks_and_every_stop_is_valid():
    vocab, pieces = _fake_vocab()
    eos = vocab.eos_id
    V = eos + 1
    piece_of = dict(enumerate(pieces))
    for schema in (FLAT, ENUM, ARRAY, OPTIONAL, NESTED):
        pda = build_json_automaton(schema)
        for seed in range(6):
            rng = random.Random(seed)
            proc = AutomatonLogitsProcessor(pda, vocab)
            proc.reset()
            seq = [777]
            out = ""
            for _ in range(500):
                masked = proc(torch.tensor([seq]), torch.zeros(1, V))
                allowed = (masked[0] != float("-inf")).nonzero().flatten().tolist()
                assert allowed, f"deadlock on {out!r}"  # invariant 1
                if eos in allowed:
                    # invariant 2: any legal stop is already schema-valid
                    jsonschema.validate(json.loads(out), schema)
                    if rng.random() < 0.6:
                        break
                    allowed = [t for t in allowed if t != eos]
                choice = rng.choice(allowed)
                out += piece_of[choice]
                seq.append(choice)
