"""Pushdown automaton (Phase 2) built as a Recursive Transition Network.

A fixed JSON schema has *bounded* nesting, so it is technically regular — but
compiling deep nesting into one flat DFA explodes, and it would not generalise
to the recursive EBNF grammars of Phase 3. So we use a real stack machine: the
grammar is a set of named rules, a ``Call`` edge pushes a return point and jumps
into a sub-rule, and reaching a rule's exit pops back. The stack is exactly the
"nested braces/brackets" depth the guide describes.

Crucially, nesting is driven by *grammar structure*, not by scanning for brace
characters — so a ``{`` inside a string literal is just an ordinary character in
the STRING rule and never touches the stack. That is what makes the hard
"token straddling a structural boundary" case fall out correctly.

The automaton exposes the same surface as the Phase-1 DFA (``start``, ``step``,
``is_dead``, ``is_live``, ``is_accepting``), so :mod:`engine.token_bridge` and
:mod:`engine.masker` drive it unchanged. A "state" here is a frozenset of
``(node, stack)`` configurations — the nondeterministic PDA simulated as a set,
the pushdown analogue of subset construction.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from engine.automaton import OTHER


# --------------------------------------------------------------------------- #
# Grammar combinators
# --------------------------------------------------------------------------- #
@dataclass
class Chars:
    """Consume one character described by ``spec``.

    spec is one of: ("lit", frozenset), ("set", frozenset), ("any",),
    ("anybut", frozenset)."""

    spec: tuple


@dataclass
class Lit:
    """Match an exact string (a sequence of single-character matches)."""

    text: str


@dataclass
class Seq:
    parts: list


@dataclass
class Alt:
    options: list


@dataclass
class Opt:
    node: object


@dataclass
class Star:
    node: object


@dataclass
class Plus:
    node: object


@dataclass
class Empty:
    pass


@dataclass
class Call:
    """Enter a named rule, pushing a return point (the stack-using edge)."""

    rule: str


@dataclass
class Grammar:
    rules: Dict[str, object]
    start: str


# Convenience constructors for common character matchers.
def lit_char(c: str) -> Chars:
    return Chars(("lit", frozenset([c])))


def char_set(chars) -> Chars:
    return Chars(("set", frozenset(chars)))


def any_char() -> Chars:
    return Chars(("any",))


def any_but(chars) -> Chars:
    return Chars(("anybut", frozenset(chars)))


# --------------------------------------------------------------------------- #
# Compilation: combinators -> RTN edges
# --------------------------------------------------------------------------- #
# Edge encodings (stored per source state):
#   ("char", spec_or_frozenset, target)  consume one char
#   ("eps", target)                      epsilon move
#   ("call", callee_entry, return_state) push return_state, enter callee
class _Builder:
    def __init__(self):
        self.edges: Dict[int, list] = defaultdict(list)
        self._n = 0
        self.alphabet: Set[str] = set()
        self.returns: Set[int] = set()
        self.rule_entry: Dict[str, int] = {}
        self.rule_exit: Dict[str, int] = {}
        self._char_edges: List[Tuple[int, int]] = []  # (state, index) to resolve

    def new_state(self) -> int:
        s = self._n
        self._n += 1
        return s

    def _note(self, chars) -> None:
        self.alphabet.update(chars)

    def add_char(self, src: int, spec: tuple, dst: int) -> None:
        # Record concrete characters so the global alphabet is complete before
        # "any"/"anybut" specs are resolved against it.
        kind = spec[0]
        if kind in ("lit", "set"):
            self._note(spec[1])
        elif kind == "anybut":
            self._note(spec[1])
        self.edges[src].append(["char", spec, dst])
        self._char_edges.append((src, len(self.edges[src]) - 1))

    def add_eps(self, src: int, dst: int) -> None:
        self.edges[src].append(("eps", dst))

    def add_call(self, src: int, entry: int, ret: int) -> None:
        self.edges[src].append(("call", entry, ret))

    # -- compile a combinator node to (entry, exit) --------------------------
    def compile(self, node) -> Tuple[int, int]:
        if isinstance(node, Empty):
            e, x = self.new_state(), self.new_state()
            self.add_eps(e, x)
            return e, x
        if isinstance(node, Chars):
            e, x = self.new_state(), self.new_state()
            self.add_char(e, node.spec, x)
            return e, x
        if isinstance(node, Lit):
            if node.text == "":
                return self.compile(Empty())
            e = self.new_state()
            cur = e
            for ch in node.text:
                nxt = self.new_state()
                self.add_char(cur, ("lit", frozenset([ch])), nxt)
                cur = nxt
            return e, cur
        if isinstance(node, Seq):
            if not node.parts:
                return self.compile(Empty())
            entry = prev_exit = None
            for part in node.parts:
                pe, px = self.compile(part)
                if entry is None:
                    entry = pe
                else:
                    self.add_eps(prev_exit, pe)
                prev_exit = px
            return entry, prev_exit
        if isinstance(node, Alt):
            e, x = self.new_state(), self.new_state()
            for opt in node.options:
                oe, ox = self.compile(opt)
                self.add_eps(e, oe)
                self.add_eps(ox, x)
            return e, x
        if isinstance(node, Opt):
            e, x = self.new_state(), self.new_state()
            ne, nx = self.compile(node.node)
            self.add_eps(e, ne)
            self.add_eps(nx, x)
            self.add_eps(e, x)  # skip
            return e, x
        if isinstance(node, Star):
            e, x = self.new_state(), self.new_state()
            ne, nx = self.compile(node.node)
            self.add_eps(e, ne)
            self.add_eps(e, x)  # zero times
            self.add_eps(nx, ne)  # loop
            self.add_eps(nx, x)  # stop
            return e, x
        if isinstance(node, Plus):
            e, x = self.new_state(), self.new_state()
            ne, nx = self.compile(node.node)
            self.add_eps(e, ne)
            self.add_eps(nx, ne)  # loop
            self.add_eps(nx, x)  # stop (at least once)
            return e, x
        if isinstance(node, Call):
            e, x = self.new_state(), self.new_state()
            self.add_call(e, self.rule_entry[node.rule], x)
            return e, x
        raise TypeError(f"unknown grammar node {node!r}")

    def resolve_specs(self) -> None:
        """Turn char specs into concrete symbol frozensets now the alphabet is
        final (``any``/``anybut`` need the whole alphabet to be known)."""
        for state, idx in self._char_edges:
            edge = self.edges[state][idx]
            spec = edge[1]
            kind = spec[0]
            if kind == "lit" or kind == "set":
                syms = frozenset(spec[1])
            elif kind == "any":
                syms = frozenset(set(self.alphabet) | {OTHER})
            elif kind == "anybut":
                syms = frozenset((set(self.alphabet) - spec[1]) | {OTHER})
            else:  # pragma: no cover
                raise ValueError(f"bad char spec {spec!r}")
            edge[1] = syms


def compile_grammar(grammar: Grammar) -> "PDA":
    b = _Builder()
    # Pre-register entry/exit states so Call can reference rules not yet built.
    for name in grammar.rules:
        b.rule_entry[name] = b.new_state()
        b.rule_exit[name] = b.new_state()
    for name, body in grammar.rules.items():
        be, bx = b.compile(body)
        b.add_eps(b.rule_entry[name], be)
        b.add_eps(bx, b.rule_exit[name])
        b.returns.add(b.rule_exit[name])
    start = b.new_state()
    accept = b.new_state()
    b.add_call(start, b.rule_entry[grammar.start], accept)
    b.resolve_specs()
    return PDA(
        edges=dict(b.edges),
        start_state=start,
        accept_state=accept,
        returns=b.returns,
        alphabet=set(b.alphabet),
    )


# --------------------------------------------------------------------------- #
# The automaton
# --------------------------------------------------------------------------- #
Config = Tuple[int, Tuple[int, ...]]  # (node, stack of return states)

_MAX_STACK = 400  # guard against pathological recursion in epsilon-closure


class PDA:
    """Nondeterministic pushdown automaton, simulated as a set of configurations.

    A "state" (what the bridge passes around) is a frozenset of ``(node, stack)``
    configurations, already epsilon-closed. Empty set == dead."""

    def __init__(self, edges, start_state, accept_state, returns, alphabet):
        self.edges = edges
        self.accept_state = accept_state
        self.returns = returns
        self.alphabet = set(alphabet)
        # Memoize (state, symbol) -> next_state. step() is a pure function of the
        # (already-closed) state and the symbol, so caching is exact. This turns
        # the bridge's per-step O(vocab) scan from ~150k closure computations
        # into ~|distinct chars| of them. Phase 3's per-state token index builds
        # on the same idea at the whole-token level.
        self._step_cache: Dict[Tuple[FrozenSet[Config], object], FrozenSet[Config]] = {}
        self.start: FrozenSet[Config] = self._closure([(start_state, ())])

    # -- interface expected by token_bridge / masker -------------------------
    def _symbol(self, ch: str):
        return ch if ch in self.alphabet else OTHER

    def step(self, state: FrozenSet[Config], ch: str) -> FrozenSet[Config]:
        sym = self._symbol(ch)
        key = (state, sym)
        cached = self._step_cache.get(key)
        if cached is not None:
            return cached
        nxt: Set[Config] = set()
        for (node, stack) in state:
            for edge in self.edges.get(node, ()):
                if edge[0] == "char" and sym in edge[1]:
                    nxt.add((edge[2], stack))
        result = self._closure(nxt)
        self._step_cache[key] = result
        return result

    def is_dead(self, state: FrozenSet[Config]) -> bool:
        return len(state) == 0

    def is_live(self, state: FrozenSet[Config]) -> bool:
        # Every reachable configuration in this grammar can be completed to
        # acceptance (the grammar has no dead-end states), so a non-empty set is
        # live. Termination of nested structure is covered by tests.
        return len(state) > 0

    def is_accepting(self, state: FrozenSet[Config]) -> bool:
        return (self.accept_state, ()) in state

    def run(self, state: FrozenSet[Config], text: str) -> FrozenSet[Config]:
        for ch in text:
            state = self.step(state, ch)
            if self.is_dead(state):
                return state
        return state

    def accepts(self, text: str) -> bool:
        """True if ``text`` is a complete sentence of the grammar."""
        return self.is_accepting(self.run(self.start, text))

    # -- epsilon / call / return closure -------------------------------------
    def _closure(self, configs) -> FrozenSet[Config]:
        seen: Set[Config] = set()
        work = list(configs)
        while work:
            cfg = work.pop()
            if cfg in seen:
                continue
            seen.add(cfg)
            node, stack = cfg
            for edge in self.edges.get(node, ()):
                if edge[0] == "eps":
                    work.append((edge[1], stack))
                elif edge[0] == "call":
                    if len(stack) < _MAX_STACK:
                        work.append((edge[1], stack + (edge[2],)))
            # Rule exit: pop and continue at the saved return point.
            if node in self.returns and stack:
                work.append((stack[-1], stack[:-1]))
        return frozenset(seen)
