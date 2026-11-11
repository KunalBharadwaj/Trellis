"""Finite automata: NFA with epsilon moves, Thompson fragments, and subset
construction to a DFA.

Design note — the OTHER sentinel
--------------------------------
A regex ranges over an unbounded character set, but a DFA needs a finite
alphabet. We collect the *concrete* characters named in the pattern into an
explicit ``alphabet`` and add one sentinel symbol :data:`OTHER` standing for
"any character not in the alphabet". ``.`` and negated classes match OTHER;
literals and positive classes do not. At run time a concrete character ``c`` is
mapped to ``c`` if it is in the alphabet, else to OTHER. This lets a finite
automaton decide membership for strings over the full Unicode range.

These classes are pure Python (no torch), so the whole regex path is unit
testable without a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple


class _Other:
    """Sentinel symbol for 'any character outside the pattern's alphabet'."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):  # pragma: no cover - cosmetic
        return "OTHER"


#: Singleton sentinel (see module docstring).
OTHER = _Other()

# A symbol is either a single-character string or the OTHER sentinel.
Symbol = object

# A sentinel DFA state id meaning "dead" (no accept reachable, consumes nothing).
DEAD = -1


@dataclass
class State:
    """An NFA state. ``eps`` are epsilon targets; ``moves`` are (symbols, target)
    consuming transitions where ``symbols`` is a frozenset over the symbol
    universe (concrete chars and/or OTHER)."""

    id: int
    eps: Set[int] = field(default_factory=set)
    moves: List[Tuple[FrozenSet, int]] = field(default_factory=list)


class NFA:
    """Epsilon-NFA with a single start and single accept state (Thompson form)."""

    def __init__(self, alphabet: Set[str]):
        self.states: Dict[int, State] = {}
        self.start: int = -1
        self.accept: int = -1
        # Concrete characters named anywhere in the pattern.
        self.alphabet: Set[str] = set(alphabet)

    # -- construction helpers -------------------------------------------------
    def new_state(self) -> int:
        sid = len(self.states)
        self.states[sid] = State(sid)
        return sid

    def add_eps(self, src: int, dst: int) -> None:
        self.states[src].eps.add(dst)

    def add_move(self, src: int, symbols: FrozenSet, dst: int) -> None:
        self.states[src].moves.append((symbols, dst))

    # -- subset construction --------------------------------------------------
    def epsilon_closure(self, ids: Iterable[int]) -> FrozenSet[int]:
        stack = list(ids)
        seen: Set[int] = set(stack)
        while stack:
            s = stack.pop()
            for nxt in self.states[s].eps:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return frozenset(seen)

    def to_dfa(self) -> "DFA":
        """Subset construction. Produces a DFA over ``alphabet ∪ {OTHER}``."""
        universe: List[Symbol] = list(self.alphabet) + [OTHER]

        start_set = self.epsilon_closure([self.start])
        set_to_id: Dict[FrozenSet[int], int] = {start_set: 0}
        id_to_set: List[FrozenSet[int]] = [start_set]
        transitions: Dict[Tuple[int, Symbol], int] = {}
        worklist = [start_set]

        while worklist:
            cur = worklist.pop()
            cur_id = set_to_id[cur]
            for sym in universe:
                # NFA states reachable by consuming one occurrence of `sym`.
                targets: Set[int] = set()
                for s in cur:
                    for symbols, dst in self.states[s].moves:
                        if sym in symbols:
                            targets.add(dst)
                if not targets:
                    continue
                closure = self.epsilon_closure(targets)
                if closure not in set_to_id:
                    set_to_id[closure] = len(id_to_set)
                    id_to_set.append(closure)
                    worklist.append(closure)
                transitions[(cur_id, sym)] = set_to_id[closure]

        accept = {
            sid
            for sid, nfa_set in enumerate(id_to_set)
            if self.accept in nfa_set
        }
        return DFA(
            n_states=len(id_to_set),
            start=0,
            accept=accept,
            transitions=transitions,
            alphabet=set(self.alphabet),
        )


class DFA:
    """Deterministic automaton over ``alphabet ∪ {OTHER}``.

    The two operations the token bridge relies on are :meth:`step` (advance by a
    concrete character, returning :data:`DEAD` when the path dies) and the
    :attr:`live_states` set (states from which an accepting state is still
    reachable — used to reject tokens that would strand us)."""

    def __init__(
        self,
        n_states: int,
        start: int,
        accept: Set[int],
        transitions: Dict[Tuple[int, Symbol], int],
        alphabet: Set[str],
    ):
        self.n_states = n_states
        self.start = start
        self.accept = set(accept)
        self.transitions = transitions
        self.alphabet = set(alphabet)
        self.live_states = self._compute_live()

    # -- queries --------------------------------------------------------------
    def _symbol(self, ch: str) -> Symbol:
        return ch if ch in self.alphabet else OTHER

    def step(self, state: int, ch: str) -> int:
        """Advance one character. Returns the next state id, or :data:`DEAD`."""
        if state == DEAD:
            return DEAD
        return self.transitions.get((state, self._symbol(ch)), DEAD)

    def run(self, state: int, text: str) -> int:
        """Advance through a whole string; DEAD if any character is rejected."""
        for ch in text:
            state = self.step(state, ch)
            if state == DEAD:
                return DEAD
        return state

    def is_dead(self, state: int) -> bool:
        return state == DEAD

    def is_accepting(self, state: int) -> bool:
        return state in self.accept

    def is_live(self, state: int) -> bool:
        """True if an accepting state is still reachable from ``state``.

        A live, non-accepting state means "valid so far, keep going"; a dead
        state means the string can never be completed into a match."""
        return state in self.live_states

    def accepts(self, text: str) -> bool:
        return self.is_accepting(self.run(self.start, text))

    # -- internals ------------------------------------------------------------
    def _compute_live(self) -> Set[int]:
        """Reverse-reachability from accepting states over all transitions."""
        preds: Dict[int, Set[int]] = {s: set() for s in range(self.n_states)}
        for (src, _sym), dst in self.transitions.items():
            preds[dst].add(src)
        live: Set[int] = set(self.accept)
        stack = list(self.accept)
        while stack:
            s = stack.pop()
            for p in preds[s]:
                if p not in live:
                    live.add(p)
                    stack.append(p)
        return live
