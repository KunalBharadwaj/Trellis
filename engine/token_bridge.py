"""The char <-> token bridge: the hard part of the project.

The automaton reasons in *characters*; the model emits *tokens* (subwords), and
a single token can span several transitions or straddle a structural boundary.
For a given automaton state we must compute the set of token ids whose decoded
text keeps the automaton on a live path.

This module is deliberately automaton-agnostic: it talks to any object exposing
``step(state, ch) -> state``, ``is_dead``, ``is_live``, ``is_accepting`` and a
``start`` attribute. The Phase-1 :class:`~engine.automaton.DFA` and the Phase-2
pushdown automaton both satisfy it, so the bridge and masker are reused as-is.

Phase 1 uses the naive, correct, O(vocab) per-step scan. Phase 3 replaces the
scan with a precomputed per-state index (:mod:`engine.token_index`) for speed;
the semantics defined here are the specification that index must match.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Set, Tuple

# The Unicode replacement character marks tokens that decode to partial bytes
# (byte-level BPE fragments). We drop them from the constrained vocabulary:
# emitting one mid-pattern can never be reasoned about at the character level.
_REPLACEMENT = "�"


@dataclass
class TokenVocab:
    """Token id -> decoded text, plus the EOS id used to end generation.

    Built once per tokenizer and reused across schemas. Construct from a real
    tokenizer with :meth:`from_hf`, or from explicit pairs with :meth:`from_pairs`
    for tests that need no model."""

    pieces: Dict[int, str]
    eos_id: int
    special_ids: Set[int] = field(default_factory=set)

    @classmethod
    def from_pairs(cls, pairs: Iterable[Tuple[int, str]], eos_id: int) -> "TokenVocab":
        return cls(pieces={tid: txt for tid, txt in pairs}, eos_id=eos_id)

    @classmethod
    def from_hf(cls, tokenizer) -> "TokenVocab":
        """Decode every vocabulary token to its text.

        Special/added tokens (other than EOS) are excluded so the model can
        never satisfy the grammar by emitting a control token. Undecodable
        partial-byte tokens are excluded too (see ``_REPLACEMENT``)."""
        eos_id = tokenizer.eos_token_id
        special = set(getattr(tokenizer, "all_special_ids", []) or [])
        pieces: Dict[int, str] = {}
        vocab_size = len(tokenizer)
        for tid in range(vocab_size):
            if tid in special and tid != eos_id:
                continue
            text = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
            if text == "" or _REPLACEMENT in text:
                continue
            pieces[tid] = text
        pieces.pop(eos_id, None)  # EOS handled structurally, not as a piece
        return cls(pieces=pieces, eos_id=eos_id, special_ids=special)


def consume(automaton, state, text: str):
    """Feed every character of ``text`` from ``state``. Returns the resulting
    state, which may be the automaton's dead sentinel if the text is rejected."""
    for ch in text:
        state = automaton.step(state, ch)
        if automaton.is_dead(state):
            return state
    return state


def advance(automaton, state, text: str):
    """Advance the automaton by a chosen token's text (assumed already legal)."""
    return consume(automaton, state, text)


def allowed_token_ids(automaton, state, vocab: TokenVocab) -> Set[int]:
    """Set of token ids legal from ``state``.

    A normal token is legal iff its whole decoded text is consumable without
    dying and lands in a *live* state (one from which an accept is still
    reachable). The EOS token is legal iff ``state`` is accepting — i.e. the
    output is already complete, so stopping is valid. The returned set is never
    empty for a reachable state unless the automaton is genuinely stuck, which
    the masker treats as an error to surface rather than hide."""
    allowed: Set[int] = set()
    for tid, piece in vocab.pieces.items():
        end = consume(automaton, state, piece)
        if not automaton.is_dead(end) and automaton.is_live(end):
            allowed.add(tid)
    if automaton.is_accepting(state):
        allowed.add(vocab.eos_id)
    return allowed
