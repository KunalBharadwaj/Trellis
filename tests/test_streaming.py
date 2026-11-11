"""Streaming + early validation (Phase 3). Model-free (fake logits_fn)."""

import json
import random

import jsonschema
import pytest

from engine.json_schema import build_json_automaton
from engine.ebnf import compile_ebnf
from engine.token_bridge import TokenVocab
from engine.streaming import (
    StreamValidator,
    StreamViolation,
    stream_constrained,
)

SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
    "additionalProperties": False,
}


def test_validator_tracks_completion_and_depth():
    pda = build_json_automaton(SCHEMA)
    v = StreamValidator(pda)
    s = v.feed('{"name": "Ada"')
    assert not s.is_complete and s.depth >= 1  # inside the object
    s = v.feed(', "age": 36}')
    assert s.is_complete and s.depth == 0      # closed, complete valid value


def test_validator_flags_first_offgrammar_token():
    pda = build_json_automaton(SCHEMA)
    v = StreamValidator(pda)
    v.feed('{"name": "Ada", ')
    with pytest.raises(StreamViolation):
        v.feed('"agee": 36}')  # wrong key -> rejected the moment it diverges


def test_validator_on_dsl():
    pda = compile_ebnf('call = name "(" ")" ; name = /[a-z]+/ ;')
    v = StreamValidator(pda)
    v.feed("foo")
    assert not v.is_complete()
    v.feed("(")
    s = v.feed(")")
    assert s.is_complete


def _fake_vocab():
    singles = [chr(c) for c in range(32, 127)]
    multis = ["true", "false", "null", '"name"', '"age"', "Ada"]
    pieces = singles + multis
    return TokenVocab.from_pairs(list(enumerate(pieces)), eos_id=len(pieces)), pieces


def test_stream_constrained_yields_valid_growing_output():
    vocab, pieces = _fake_vocab()
    pda = build_json_automaton(SCHEMA)
    rng = random.Random(0)

    # Fake model: random logits, but bias EOS up once output can validly stop,
    # so the stream terminates with a complete value.
    def logits_fn(_generated):
        return [rng.random() for _ in range(vocab.eos_id + 1)]

    def sample(logits, allowed):
        allowed = list(allowed)
        if vocab.eos_id in allowed and rng.random() < 0.25:
            return vocab.eos_id
        non_eos = [t for t in allowed if t != vocab.eos_id] or allowed
        return max(non_eos, key=lambda t: logits[t])

    last = None
    prev_len = -1
    for status in stream_constrained(logits_fn, pda, vocab, max_new_tokens=400, sample=sample):
        assert len(status.text) >= prev_len  # output only grows
        prev_len = len(status.text)
        # Whenever the stream reports complete, the text really validates.
        if status.is_complete and status.text:
            jsonschema.validate(json.loads(status.text), SCHEMA)
        last = status
        if status.done:
            break
    assert last is not None and last.done
