"""EBNF frontend + tool-call DSL (Phase 3): parsing, recursion, the regex
terminal reuse, and the masking invariant. Model-free."""

import os
import random

import torch

from engine.ebnf import compile_ebnf, parse_ebnf, build_ebnf_processor
from engine.token_bridge import TokenVocab

GRAMMAR_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "grammars", "tool_call.ebnf")
TOOL_CALL = open(GRAMMAR_PATH).read()


def test_basic_ebnf_constructs():
    g = compile_ebnf('greeting = "hello" | "hi" ;')
    assert g.accepts("hello") and g.accepts("hi")
    assert not g.accepts("hey") and not g.accepts("")


def test_optional_star_plus_and_grouping():
    g = compile_ebnf('x = "a" ["b"] { "c" } ;')          # a, optional b, zero+ c
    for s in ["a", "ab", "ac", "abc", "abccc", "accc"]:
        assert g.accepts(s), s
    assert not g.accepts("b")
    g2 = compile_ebnf('x = "a"+ "b"? ;')
    for s in ["a", "aa", "ab", "aaab"]:
        assert g2.accepts(s), s
    assert not g2.accepts("b") and not g2.accepts("abb")


def test_regex_terminal_reuse():
    g = compile_ebnf("num = /-?[0-9]+/ ;")
    for s in ["0", "42", "-7"]:
        assert g.accepts(s), s
    for s in ["", "-", "4a"]:
        assert not g.accepts(s), s


def test_tool_call_dsl_accept_reject():
    pda = compile_ebnf(TOOL_CALL)
    for s in ["f()", "add(1,2)", "add(1, 2)", "neg(-4)", 'x(true,false,3.14)',
              "add(mul(2,3),neg(-4))", "f(g(h(i(1))))"]:
        assert pda.accepts(s), s
    for s in ["F()", "add(1,2", "add(,2)", "add(1,)", "1+2", "add(1 2)", "name"]:
        assert not pda.accepts(s), s


def test_recursion_uses_the_stack():
    pda = compile_ebnf(TOOL_CALL)

    def max_stack(text):
        st, m = pda.start, 0
        for ch in text:
            st = pda.step(st, ch)
            for _node, stack in st:
                m = max(m, len(stack))
        return m

    # Deeper nesting must push the stack deeper — impossible for a DFA.
    assert max_stack("f(1)") < max_stack("f(g(1))") < max_stack("f(g(h(1))))"[:-1])


def test_undefined_rule_raises():
    import pytest

    with pytest.raises(ValueError):
        parse_ebnf("a = b ;")  # b is never defined


def test_masking_invariant_on_dsl():
    pda = compile_ebnf(TOOL_CALL)
    singles = [chr(c) for c in range(32, 127)]
    multis = ["add", "mul", "neg", "true", "false", "()", "(1)", "1,2", "))", '"x"']
    pieces = singles + multis
    vocab = TokenVocab.from_pairs(list(enumerate(pieces)), eos_id=len(pieces))
    eos = vocab.eos_id
    V = eos + 1
    proc = build_ebnf_processor(TOOL_CALL, vocab)
    for seed in range(5):
        rng = random.Random(seed)
        proc.reset()
        seq = [1]
        depth_seen = 0
        for _ in range(300):
            masked = proc(torch.tensor([seq]), torch.zeros(1, V))
            allowed = (masked[0] != float("-inf")).nonzero().flatten().tolist()
            assert allowed, "deadlock"  # never stuck
            if eos in allowed and rng.random() < 0.5:
                break
            allowed = [t for t in allowed if t != eos] or allowed
            seq.append(rng.choice(allowed))
