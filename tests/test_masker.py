"""LogitsProcessor behaviour (Phase 1), driven with fake logits — no model."""

import re

import torch

from engine.regex_compiler import compile_regex
from engine.token_bridge import TokenVocab
from engine.masker import AutomatonLogitsProcessor


def build(pieces, eos_id, pattern):
    vocab = TokenVocab.from_pairs(list(enumerate(pieces)), eos_id=eos_id)
    return AutomatonLogitsProcessor(compile_regex(pattern), vocab), vocab


def allowed_ids(masked_row):
    return (masked_row != float("-inf")).nonzero().flatten().tolist()


def test_illegal_tokens_masked_to_neg_inf():
    pieces = list("0123456789abc")
    eos = len(pieces)
    V = eos + 1
    proc, _ = build(pieces, eos, r"\d{2}")
    proc.reset()
    scores = torch.zeros(1, V)
    masked = proc(torch.tensor([[7, 8]]), scores)  # 2 fake prompt tokens
    allowed = set(allowed_ids(masked[0]))
    assert allowed == {i for i, p in enumerate(pieces) if p.isdigit()}
    assert eos not in allowed


def test_never_all_neg_inf():
    pieces = list("0123456789")
    eos = len(pieces)
    V = eos + 1
    proc, _ = build(pieces, eos, r"\d{3}")
    proc.reset()
    masked = proc(torch.tensor([[1]]), torch.zeros(1, V))
    assert (masked[0] != float("-inf")).any()


def test_state_advances_across_steps():
    # Emulate the generate() call pattern: input_ids grows by one each step.
    pieces = list("0123456789")
    eos = len(pieces)
    V = eos + 1
    proc, _ = build(pieces, eos, r"\d{2}")
    proc.reset()
    seq = [99]  # one prompt token
    # Step 1
    proc(torch.tensor([seq]), torch.zeros(1, V))
    seq.append(4)  # model "chose" digit '4'
    # Step 2
    proc(torch.tensor([seq]), torch.zeros(1, V))
    seq.append(2)  # chose '2' -> now two digits, accepting
    # Step 3: EOS must now be allowed
    masked = proc(torch.tensor([seq]), torch.zeros(1, V))
    assert eos in allowed_ids(masked[0])


def test_full_greedy_loop_produces_valid_string():
    pieces = list("0123456789-") + ["20", "26", "2026", "-0"]
    eos = len(pieces)
    V = eos + 1
    proc, vocab = build(pieces, eos, r"\d{4}-\d{2}-\d{2}")
    proc.reset()
    piece_of = dict(enumerate(pieces))
    seq = [99]
    out = ""
    for _ in range(40):
        masked = proc(torch.tensor([seq]), torch.zeros(1, V))
        allowed = allowed_ids(masked[0])
        assert allowed, "deadlock"
        non_eos = [t for t in allowed if t != eos]
        choice = eos if not non_eos else max(non_eos, key=lambda t: len(piece_of[t]))
        if choice == eos:
            break
        out += piece_of[choice]
        seq.append(choice)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", out), f"invalid: {out!r}"


def test_reset_restores_start_state():
    pieces = list("ab")
    eos = len(pieces)
    V = eos + 1
    proc, _ = build(pieces, eos, r"ab")
    proc.reset()
    seq = [9]
    proc(torch.tensor([seq]), torch.zeros(1, V))
    seq.append(0)  # 'a'
    proc(torch.tensor([seq]), torch.zeros(1, V))
    # After reset, we should be back at the start expecting 'a' only.
    proc.reset()
    masked = proc(torch.tensor([[9]]), torch.zeros(1, V))
    assert allowed_ids(masked[0]) == [0]  # only 'a'
