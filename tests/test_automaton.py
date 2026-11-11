"""Regex -> NFA -> DFA correctness (Phase 1)."""

import pytest

from engine.regex_compiler import compile_regex
from engine.automaton import DEAD


@pytest.mark.parametrize(
    "pattern,accepts,rejects",
    [
        (r"\d{4}-\d{2}-\d{2}", ["2026-11-06", "0000-00-00"], ["2026-11-6", "2026/11/06", ""]),
        (r"(red|green|blue)", ["red", "green", "blue"], ["", "re", "yellow", "red "]),
        (r"[A-Z][a-z]*", ["H", "Hello", "Zoo"], ["hello", "", "H3"]),
        (r"-?\d+", ["0", "42", "-7"], ["", "-", "4-2", "x"]),
        (r"[^0-9]+", ["abc", "x y"], ["", "a1", "123"]),
        (r"a{2,3}", ["aa", "aaa"], ["a", "aaaa", ""]),
        (r"a{2,}", ["aa", "aaaa"], ["a", ""]),
        (r"v\d\.\d", ["v1.2"], ["v1x2", "v1.23"]),
        (r"a.c", ["abc", "a9c", "a-c"], ["ac", "abbc"]),
        (r"#[0-9a-fA-F]{6}", ["#00ff00", "#ABCDEF"], ["#fff", "00ff00", "#gggggg"]),
        (r"\(\d{3}\) \d{3}-\d{4}", ["(123) 456-7890"], ["123 456-7890", "(12) 456-7890"]),
    ],
)
def test_accept_reject(pattern, accepts, rejects):
    dfa = compile_regex(pattern)
    for s in accepts:
        assert dfa.accepts(s), f"{pattern!r} should accept {s!r}"
    for s in rejects:
        assert not dfa.accepts(s), f"{pattern!r} should reject {s!r}"


def test_live_vs_accepting():
    dfa = compile_regex(r"\d{4}")
    st = dfa.start
    # Start is live (can reach accept) but not accepting.
    assert dfa.is_live(st) and not dfa.is_accepting(st)
    for _ in range(4):
        st = dfa.step(st, "1")
        assert not dfa.is_dead(st)
    assert dfa.is_accepting(st)
    # A fifth digit has no transition -> dead.
    assert dfa.is_dead(dfa.step(st, "1"))


def test_empty_pattern_accepts_empty():
    dfa = compile_regex(r"")
    assert dfa.accepts("")
    assert not dfa.accepts("a")


def test_other_sentinel_handles_unseen_chars():
    # Negated class must accept characters never named in the pattern.
    dfa = compile_regex(r"[^x]")
    assert dfa.accepts("q")  # 'q' is not in the alphabet -> maps to OTHER
    assert dfa.accepts("Z")
    assert not dfa.accepts("x")


def test_alternation_of_different_lengths():
    dfa = compile_regex(r"(yes|no)")
    assert dfa.accepts("yes") and dfa.accepts("no")
    assert not dfa.accepts("y") and not dfa.accepts("ye")
