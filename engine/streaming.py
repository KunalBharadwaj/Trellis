"""Streaming output with early validation (Phase 3).

Two complementary tools over the same automaton:

* :class:`StreamValidator` — feed an *existing* token/text stream and learn, at
  each token, whether the output is still on-grammar, whether it is a complete
  valid value yet, and the current nesting depth. The first off-grammar token is
  reported immediately (fail fast), rather than only after the whole string is
  produced.

* :func:`stream_constrained` — drive a model's decode loop token by token,
  masking to the grammar and *yielding* the growing output plus status as it
  forms. Because every token is masked, the output is valid by construction; the
  status tells you the moment it becomes a complete value and how deep the
  structure currently is.

The core is model-agnostic (it takes a ``logits_fn``), so it is unit-tested
without a model; :func:`stream_generate_hf` is a thin Hugging Face adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional

from engine.token_bridge import TokenVocab, advance, allowed_token_ids


def _depth(state) -> int:
    """Current nesting depth, as the *shallowest* live PDA stack (0 for a DFA).

    With the PDA simulated as a set of configurations, several parse paths
    coexist; the shallowest is the one nearest to completing, so it is 0 exactly
    when the output is a complete top-level value and grows as structure opens."""
    try:
        return min((len(stack) for _node, stack in state), default=0)
    except TypeError:
        return 0


@dataclass
class StreamStatus:
    text: str          # full output so far
    piece: str         # the text just added
    is_complete: bool  # output so far is a complete valid value (automaton accepting)
    depth: int         # current structural nesting depth
    n_tokens: int
    done: bool = False  # generation finished (EOS or budget)


# --------------------------------------------------------------------------- #
# Validate an existing stream
# --------------------------------------------------------------------------- #
class StreamValidator:
    """Incrementally validate a stream against an automaton.

    Feed decoded pieces (per token, or any chunk). Returns a :class:`StreamStatus`
    where ``is_complete`` says the output is a complete valid value so far. If a
    piece leaves the grammar, :meth:`feed` raises :class:`StreamViolation` naming
    the offending text — the early-rejection signal."""

    def __init__(self, automaton):
        self.automaton = automaton
        self.state = automaton.start
        self.text = ""
        self.n = 0

    def feed(self, piece: str) -> StreamStatus:
        state = self.state
        for i, ch in enumerate(piece):
            state = self.automaton.step(state, ch)
            if self.automaton.is_dead(state):
                raise StreamViolation(self.text + piece[: i + 1], piece, i)
        self.state = state
        self.text += piece
        self.n += 1
        return StreamStatus(
            text=self.text, piece=piece,
            is_complete=self.automaton.is_accepting(state),
            depth=_depth(state), n_tokens=self.n,
        )

    def is_complete(self) -> bool:
        return self.automaton.is_accepting(self.state)


class StreamViolation(ValueError):
    def __init__(self, prefix: str, piece: str, offset: int):
        self.prefix = prefix
        self.piece = piece
        self.offset = offset
        super().__init__(
            f"stream left the grammar at {prefix[-40:]!r} "
            f"(offending char {piece[offset]!r} in piece {piece!r})"
        )


# --------------------------------------------------------------------------- #
# Constrained streaming generation (model-agnostic core)
# --------------------------------------------------------------------------- #
def _greedy(logits, allowed_ids) -> int:
    best, best_v = None, None
    for tid in allowed_ids:
        v = logits[tid]
        if best_v is None or v > best_v:
            best, best_v = tid, v
    return best


def stream_constrained(
    logits_fn: Callable[[List[int]], "object"],
    automaton,
    vocab: TokenVocab,
    *,
    allowed_fn=None,
    max_new_tokens: int = 256,
    sample: Optional[Callable[["object", set], int]] = None,
) -> Iterator[StreamStatus]:
    """Yield a :class:`StreamStatus` after each generated token.

    ``logits_fn(generated_ids)`` returns next-token scores (indexable by id).
    ``sample(logits, allowed_ids)`` picks a token id from the allowed set;
    defaults to greedy argmax over the allowed set."""
    allowed_fn = allowed_fn or allowed_token_ids
    choose = sample or _greedy
    state = automaton.start
    generated: List[int] = []
    text = ""
    for _ in range(max_new_tokens):
        allowed = allowed_fn(automaton, state, vocab)
        if not allowed:
            raise RuntimeError("stream_constrained: no legal token and no legal stop")
        logits = logits_fn(generated)
        tid = choose(logits, allowed)
        if tid == vocab.eos_id:
            yield StreamStatus(text=text, piece="", is_complete=True,
                               depth=_depth(state), n_tokens=len(generated), done=True)
            return
        piece = vocab.pieces[tid]
        state = advance(automaton, state, piece)
        generated.append(tid)
        text += piece
        yield StreamStatus(text=text, piece=piece,
                           is_complete=automaton.is_accepting(state),
                           depth=_depth(state), n_tokens=len(generated), done=False)
    # Budget exhausted.
    yield StreamStatus(text=text, piece="", is_complete=automaton.is_accepting(state),
                       depth=_depth(state), n_tokens=len(generated), done=True)


def stream_generate_hf(model, tokenizer, prompt: str, automaton, vocab: TokenVocab,
                       *, max_new_tokens: int = 256, **kw) -> Iterator[StreamStatus]:
    """Hugging Face adapter: stream constrained generation for ``prompt``.

    Accepts either a raw HF model or this project's ``harness.Model`` wrapper
    (unwrapped via its ``model_id`` marker, so a real HF model's own ``.model``
    attribute is never mistaken for the wrapper). Demo-grade loop: re-runs the
    prompt+generated prefix each step — fine for the short outputs here."""
    import torch

    hf_model = model.model if (hasattr(model, "model_id") and hasattr(model, "model")) else model
    device = getattr(hf_model, "device", "cpu")
    base = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    def logits_fn(generated_ids):
        if generated_ids:
            extra = torch.tensor([generated_ids], device=device)
            ids = torch.cat([base, extra], dim=1)
        else:
            ids = base
        with torch.no_grad():
            out = hf_model(ids)
        return out.logits[0, -1]

    yield from stream_constrained(logits_fn, automaton, vocab,
                                  max_new_tokens=max_new_tokens, **kw)
