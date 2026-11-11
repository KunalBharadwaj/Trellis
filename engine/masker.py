"""The LogitsProcessor that enforces the grammar at decode time.

On every step it (1) advances the automaton by whatever token(s) the model just
produced, (2) derives the legal token set from the current state via the token
bridge, and (3) sets every illegal token's logit to -inf so sampling can only
pick a legal one. Output can therefore never leave the grammar.

Invariants (from the guide's gotchas):
  * Never mask *every* token to -inf. When the state is accepting, EOS is always
    legal, so the model can stop. A state with no legal continuation and no legal
    stop is an automaton bug, surfaced as an error rather than silently hung.
  * EOS is legal exactly when the automaton is in an accepting state.

Batching: the benchmark harness decodes one sequence at a time (batch size 1),
which is what this processor supports. Multi-beam/multi-sequence constrained
decoding needs per-beam state and is out of scope for the harness.
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers import LogitsProcessor

from engine.token_bridge import TokenVocab, advance, allowed_token_ids


class AutomatonLogitsProcessor(LogitsProcessor):
    """Constrain generation to an automaton. Works with any automaton exposing
    ``start``, ``step``, ``is_dead``, ``is_live``, ``is_accepting`` (the DFA and
    the pushdown automaton both do)."""

    def __init__(self, automaton, vocab: TokenVocab, allowed_fn=allowed_token_ids):
        self.automaton = automaton
        self.vocab = vocab
        self._allowed_fn = allowed_fn
        self.reset()

    def reset(self) -> None:
        """Reset to the start state. Call between independent generations."""
        self.state = self.automaton.start
        self._prompt_len: Optional[int] = None
        self._consumed = 0

    # -- state tracking -------------------------------------------------------
    def _sync_state(self, seq: torch.Tensor) -> None:
        """Advance the automaton by any tokens generated since the last call."""
        if self._prompt_len is None:
            # First call: everything present is the prompt, nothing generated.
            self._prompt_len = int(seq.shape[0])
            return
        new_start = self._prompt_len + self._consumed
        for tid in seq[new_start:].tolist():
            if tid == self.vocab.eos_id:
                continue  # EOS ends generation; it consumes no grammar characters
            piece = self.vocab.pieces.get(tid, "")
            if piece:
                self.state = advance(self.automaton, self.state, piece)
            self._consumed += 1

    # -- the hook -------------------------------------------------------------
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        seq = input_ids[0]
        self._sync_state(seq)

        allowed = self._allowed_fn(self.automaton, self.state, self.vocab)
        if not allowed:
            raise RuntimeError(
                "constrained decoding reached a state with no legal token and no "
                "legal stop (state=%r). This is an automaton/bridge bug, not an "
                "acceptable output." % (self.state,)
            )

        idx = torch.tensor(sorted(allowed), device=scores.device, dtype=torch.long)
        masked = torch.full_like(scores, float("-inf"))
        masked[:, idx] = scores[:, idx]
        return masked


def build_regex_processor(
    pattern: str, vocab: TokenVocab, indexed: bool = False, precompute: bool = False
) -> AutomatonLogitsProcessor:
    """Convenience: compile ``pattern`` and wrap it in a processor.

    ``indexed`` adds the Phase-3 per-state token cache; ``precompute`` fills it
    eagerly up front (cheap for a DFA)."""
    from engine.regex_compiler import compile_regex

    automaton = compile_regex(pattern)
    allowed_fn = allowed_token_ids
    if indexed:
        from engine.token_index import TokenIndex

        idx = TokenIndex(automaton, vocab)
        if precompute:
            idx.precompute()
        allowed_fn = idx
    return AutomatonLogitsProcessor(automaton, vocab, allowed_fn=allowed_fn)
