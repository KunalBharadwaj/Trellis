"""Precomputed per-state token-mask index (Phase 3) — the speed fix.

The Phase-1/2 bridge recomputes the allowed-token set by scanning the whole
vocabulary (~150k tokens) at *every* decode step. But ``allowed_token_ids`` is a
pure function of the automaton state, so the result can be cached by state: the
scan then happens once per *distinct state* instead of once per step, and
decode-time masking becomes a dictionary lookup.

``TokenIndex`` is a drop-in for the masker's ``allowed_fn`` (same call
signature), so nothing else changes. It fills lazily on first visit, or eagerly
via :meth:`precompute`, which walks the reachable state graph once up front.

Correctness is unchanged — the cached set is exactly what the naive scan would
return (a test asserts this). Only latency differs.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Optional, Set

from engine.token_bridge import TokenVocab, allowed_token_ids


class TokenIndex:
    """Per-state cache of allowed token ids.

    Usage::

        index = TokenIndex(automaton, vocab)
        proc = AutomatonLogitsProcessor(automaton, vocab, allowed_fn=index)
    """

    def __init__(self, automaton, vocab: TokenVocab):
        self.automaton = automaton
        self.vocab = vocab
        self._cache: Dict[object, FrozenSet[int]] = {}
        self.hits = 0
        self.misses = 0

    # Matches the allowed_fn signature (automaton, state, vocab); the passed
    # automaton/vocab are ignored in favour of the ones this index was built for.
    def __call__(self, automaton, state, vocab) -> Set[int]:
        cached = self._cache.get(state)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        result = frozenset(allowed_token_ids(self.automaton, state, self.vocab))
        self._cache[state] = result
        return result

    def allowed(self, state) -> Set[int]:
        return self(self.automaton, state, self.vocab)

    def precompute(self, max_states: int = 100_000) -> "TokenIndex":
        """Eagerly fill the cache by walking every reachable state.

        Breadth-first over the automaton's alphabet (plus one representative
        out-of-alphabet character, to cover the OTHER sentinel's transitions).
        Bounded by ``max_states`` so a pathological grammar can't run away."""
        aut = self.automaton
        # A character that is guaranteed to map to OTHER (not in the alphabet).
        other_probe = None
        for cand in ("\x00", "￿", "\x01"):
            if cand not in aut.alphabet:
                other_probe = cand
                break
        probe_chars = list(aut.alphabet)
        if other_probe is not None:
            probe_chars.append(other_probe)

        seen: Set[object] = set()
        frontier = [aut.start]
        seen.add(aut.start)
        while frontier:
            if len(seen) > max_states:
                raise RuntimeError(
                    f"precompute exceeded {max_states} states; grammar may be "
                    "unbounded — use lazy indexing instead."
                )
            state = frontier.pop()
            self(aut, state, self.vocab)  # compute + cache allowed set
            for ch in probe_chars:
                nxt = aut.step(state, ch)
                if not aut.is_dead(nxt) and nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
        return self

    @property
    def n_states(self) -> int:
        return len(self._cache)
