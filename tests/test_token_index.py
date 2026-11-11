"""Per-state token index (Phase 3): identical results to the naive scan, plus
eager precompute. Model-free."""

import random

import torch

from engine.regex_compiler import compile_regex
from engine.json_schema import build_json_automaton, build_json_processor
from engine.token_bridge import TokenVocab, allowed_token_ids, advance
from engine.token_index import TokenIndex
from engine.masker import AutomatonLogitsProcessor


def _vocab():
    singles = [chr(c) for c in range(32, 127)]
    multis = ["true", "false", "null", '"name"', '"age"', "Ada", '{"', '":', '",']
    pieces = singles + multis
    return TokenVocab.from_pairs(list(enumerate(pieces)), eos_id=len(pieces)), pieces


SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
    "additionalProperties": False,
}


def test_index_matches_naive_along_a_path():
    vocab, _ = _vocab()
    pda = build_json_automaton(SCHEMA)
    index = TokenIndex(pda, vocab)
    state = pda.start
    for ch in '{"name": "Ada", "age": 36}':
        assert index.allowed(state) == allowed_token_ids(pda, state, vocab)
        state = advance(pda, state, ch)


def test_cache_is_used_on_revisit():
    vocab, _ = _vocab()
    pda = build_json_automaton(SCHEMA)
    index = TokenIndex(pda, vocab)
    s = pda.start
    index.allowed(s)
    before = index.misses
    index.allowed(s)  # same state again
    assert index.hits >= 1 and index.misses == before


def test_precompute_fills_cache_and_matches():
    vocab, _ = _vocab()
    dfa = compile_regex(r"\d{4}-\d{2}-\d{2}")
    index = TokenIndex(dfa, vocab).precompute()
    assert index.n_states > 0
    # Every state visited during a run is already cached (no new misses).
    misses_after_precompute = index.misses
    state = dfa.start
    for ch in "2026-11-11":
        index.allowed(state)
        state = advance(dfa, state, ch)
    assert index.misses == misses_after_precompute  # all hits


def test_indexed_processor_decodes_identically():
    vocab, pieces = _vocab()
    eos = vocab.eos_id
    V = eos + 1
    piece_of = dict(enumerate(pieces))

    def run(indexed):
        proc = build_json_processor(SCHEMA, vocab, indexed=indexed)
        proc.reset()
        rng = random.Random(0)
        seq = [1]
        out = ""
        for _ in range(400):
            masked = proc(torch.tensor([seq]), torch.zeros(1, V))
            allowed = (masked[0] != float("-inf")).nonzero().flatten().tolist()
            assert allowed
            if eos in allowed and rng.random() < 0.5:
                break
            allowed = [t for t in allowed if t != eos] or allowed
            choice = rng.choice(allowed)
            if choice == eos:
                break
            out += piece_of[choice]
            seq.append(choice)
        return out

    # Same seed + same allowed sets => identical token choices.
    assert run(False) == run(True)
