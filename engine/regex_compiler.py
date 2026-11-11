"""Regex -> DFA, hand-rolled.

Pipeline: recursive-descent parse into an AST, Thompson construction into an
epsilon-NFA, then subset construction (:meth:`NFA.to_dfa`) into a DFA.

Supported subset (enough for dates, enums, fixed formats, and JSON scalars):
    literals and escapes  \\. \\d \\w \\s \\D \\W \\S \\n \\t \\r and \\<meta>
    . (any character)     char classes [abc] [a-z] [^0-9] with ranges
    quantifiers           *  +  ?  {m}  {m,}  {m,n}
    grouping              ( ... )        alternation  a|b
Concatenation is implicit. Anchors, backreferences, and lookaround are out of
scope (documented in the README).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set

from engine.automaton import NFA, OTHER, DFA


# --------------------------------------------------------------------------- #
# AST
# --------------------------------------------------------------------------- #
@dataclass
class _Spec:
    """A one-symbol matcher resolved against the final alphabet at build time."""

    kind: str  # "lit" | "any" | "class"
    chars: Set[str]  # for "lit"/"class": the named characters
    negated: bool = False  # for "class"

    def symbols(self, alphabet: Set[str]):
        if self.kind == "lit":
            return frozenset(self.chars)
        if self.kind == "any":
            return frozenset(set(alphabet) | {OTHER})
        # class
        if not self.negated:
            return frozenset(self.chars)
        return frozenset((set(alphabet) - self.chars) | {OTHER})


@dataclass
class _Leaf:
    spec: _Spec


@dataclass
class _Empty:
    pass


@dataclass
class _Concat:
    parts: List[object]


@dataclass
class _Alt:
    options: List[object]


@dataclass
class _Repeat:
    node: object
    m: int
    n: Optional[int]  # None = unbounded


# Character-class expansions for escapes.
_DIGITS = set("0123456789")
_WORD = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
_SPACE = set(" \t\n\r\f\v")


class _Parser:
    """Recursive-descent parser. Collects concrete characters into ``alphabet``
    so the automaton's symbol universe is known before NFA construction."""

    def __init__(self, pattern: str):
        self.s = pattern
        self.i = 0
        self.alphabet: Set[str] = set()

    # -- low-level ------------------------------------------------------------
    def _peek(self) -> Optional[str]:
        return self.s[self.i] if self.i < len(self.s) else None

    def _next(self) -> str:
        ch = self.s[self.i]
        self.i += 1
        return ch

    def _eof(self) -> bool:
        return self.i >= len(self.s)

    def _note(self, chars) -> None:
        self.alphabet.update(chars)

    # -- grammar: alt -> concat ('|' concat)* ---------------------------------
    def parse(self):
        node = self._alt()
        if not self._eof():
            raise ValueError(f"unexpected {self._peek()!r} at position {self.i}")
        return node

    def _alt(self):
        options = [self._concat()]
        while self._peek() == "|":
            self._next()
            options.append(self._concat())
        return options[0] if len(options) == 1 else _Alt(options)

    def _concat(self):
        parts = []
        while True:
            ch = self._peek()
            if ch is None or ch in "|)":
                break
            parts.append(self._repeat())
        if not parts:
            return _Empty()
        return parts[0] if len(parts) == 1 else _Concat(parts)

    def _repeat(self):
        atom = self._atom()
        ch = self._peek()
        if ch == "*":
            self._next()
            return _Repeat(atom, 0, None)
        if ch == "+":
            self._next()
            return _Repeat(atom, 1, None)
        if ch == "?":
            self._next()
            return _Repeat(atom, 0, 1)
        if ch == "{":
            return self._brace(atom)
        return atom

    def _brace(self, atom):
        assert self._next() == "{"
        digits = ""
        while self._peek() is not None and self._peek().isdigit():
            digits += self._next()
        m = int(digits) if digits else 0
        n: Optional[int] = m
        if self._peek() == ",":
            self._next()
            d2 = ""
            while self._peek() is not None and self._peek().isdigit():
                d2 += self._next()
            n = int(d2) if d2 else None
        if self._peek() != "}":
            raise ValueError(f"unterminated {{}} quantifier at position {self.i}")
        self._next()  # consume }
        return _Repeat(atom, m, n)

    def _atom(self):
        ch = self._peek()
        if ch == "(":
            self._next()
            node = self._alt()
            if self._peek() != ")":
                raise ValueError("unbalanced parenthesis")
            self._next()
            return node
        if ch == "[":
            return self._char_class()
        if ch == ".":
            self._next()
            return _Leaf(_Spec("any", set()))
        if ch == "\\":
            return self._escape()
        if ch in ")|*+?{}":
            raise ValueError(f"unexpected {ch!r} at position {self.i}")
        # literal
        self._next()
        self._note([ch])
        return _Leaf(_Spec("lit", {ch}))

    def _escape(self):
        assert self._next() == "\\"
        if self._eof():
            raise ValueError("trailing backslash")
        ch = self._next()
        mapping = {
            "d": ("class", _DIGITS, False),
            "D": ("class", _DIGITS, True),
            "w": ("class", _WORD, False),
            "W": ("class", _WORD, True),
            "s": ("class", _SPACE, False),
            "S": ("class", _SPACE, True),
        }
        if ch in mapping:
            kind, chars, neg = mapping[ch]
            self._note(chars)
            return _Leaf(_Spec("class", set(chars), neg))
        literal = {"n": "\n", "t": "\t", "r": "\r", "f": "\f", "v": "\v", "0": "\0"}
        real = literal.get(ch, ch)  # \<meta> -> the metacharacter literally
        self._note([real])
        return _Leaf(_Spec("lit", {real}))

    def _char_class(self):
        assert self._next() == "["
        negated = False
        if self._peek() == "^":
            self._next()
            negated = True
        chars: Set[str] = set()
        if self._peek() == "]":  # literal ] as first member
            chars.add(self._next())
        while self._peek() is not None and self._peek() != "]":
            c = self._class_char()
            # range a-z
            if self._peek() == "-" and self.i + 1 < len(self.s) and self.s[self.i + 1] != "]":
                self._next()  # consume '-'
                end = self._class_char()
                for code in range(ord(c), ord(end) + 1):
                    chars.add(chr(code))
            else:
                chars.add(c)
        if self._peek() != "]":
            raise ValueError("unterminated character class")
        self._next()  # consume ]
        self._note(chars)
        return _Leaf(_Spec("class", chars, negated))

    def _class_char(self) -> str:
        ch = self._next()
        if ch == "\\":
            if self._eof():
                raise ValueError("trailing backslash in class")
            e = self._next()
            return {"n": "\n", "t": "\t", "r": "\r", "f": "\f", "v": "\v"}.get(e, e)
        return ch


# --------------------------------------------------------------------------- #
# Thompson construction: AST -> NFA fragment
# --------------------------------------------------------------------------- #
def _build(node, nfa: NFA):
    """Return (start, accept) for ``node``, adding states to ``nfa``.

    Called repeatedly on the same subtree for ``{m,n}`` expansion, each call
    minting fresh states — so copies share no state."""
    if isinstance(node, _Empty):
        s, a = nfa.new_state(), nfa.new_state()
        nfa.add_eps(s, a)
        return s, a

    if isinstance(node, _Leaf):
        s, a = nfa.new_state(), nfa.new_state()
        nfa.add_move(s, node.spec.symbols(nfa.alphabet), a)
        return s, a

    if isinstance(node, _Concat):
        start = prev_accept = None
        for part in node.parts:
            s, a = _build(part, nfa)
            if start is None:
                start = s
            else:
                nfa.add_eps(prev_accept, s)
            prev_accept = a
        return start, prev_accept

    if isinstance(node, _Alt):
        s, a = nfa.new_state(), nfa.new_state()
        for opt in node.options:
            os_, oa = _build(opt, nfa)
            nfa.add_eps(s, os_)
            nfa.add_eps(oa, a)
        return s, a

    if isinstance(node, _Repeat):
        return _build_repeat(node, nfa)

    raise TypeError(f"unknown AST node {node!r}")


def _build_repeat(node: _Repeat, nfa: NFA):
    m, n = node.m, node.n
    start = nfa.new_state()
    cur = start
    # `m` mandatory copies.
    for _ in range(m):
        s, a = _build(node.node, nfa)
        nfa.add_eps(cur, s)
        cur = a
    if n is None:
        # Unbounded tail: a Kleene star over one more copy.
        loop_start = nfa.new_state()
        nfa.add_eps(cur, loop_start)
        s, a = _build(node.node, nfa)
        nfa.add_eps(loop_start, s)
        nfa.add_eps(a, loop_start)
        accept = nfa.new_state()
        nfa.add_eps(loop_start, accept)
        return start, accept
    # Bounded tail: (n - m) optional copies, each skippable to the end.
    accept = nfa.new_state()
    nfa.add_eps(cur, accept)
    for _ in range(n - m):
        s, a = _build(node.node, nfa)
        nfa.add_eps(cur, s)
        nfa.add_eps(a, accept)
        cur = a
    return start, accept


def compile_regex(pattern: str) -> DFA:
    """Compile ``pattern`` (a supported-subset regex) into a :class:`DFA`."""
    parser = _Parser(pattern)
    ast = parser.parse()
    nfa = NFA(parser.alphabet)
    start, accept = _build(ast, nfa)
    nfa.start, nfa.accept = start, accept
    return nfa.to_dfa()


def parse_regex_ast(pattern: str):
    """Parse ``pattern`` and return ``(ast, alphabet)``.

    Exposes the regex AST so other compilers (e.g. the Phase-3 EBNF frontend)
    can translate a ``/regex/`` terminal into pushdown grammar combinators,
    reusing this parser instead of a second regex implementation."""
    parser = _Parser(pattern)
    ast = parser.parse()
    return ast, parser.alphabet
