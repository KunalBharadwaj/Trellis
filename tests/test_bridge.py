"""Char <-> token bridge (Phase 1): the hard part, tested without a model."""

from engine.regex_compiler import compile_regex
from engine.token_bridge import TokenVocab, allowed_token_ids, consume, advance


def make_vocab(pieces, eos_id):
    """pieces: list of strings; ids are their indices; eos_id separate."""
    return TokenVocab.from_pairs(list(enumerate(pieces)), eos_id=eos_id)


def test_single_char_tokens_allowed_only_when_legal():
    dfa = compile_regex(r"\d{2}")
    pieces = list("0123456789abc")
    eos = len(pieces)
    vocab = make_vocab(pieces, eos)
    allowed = allowed_token_ids(dfa, dfa.start, vocab)
    # digits allowed, letters not, EOS not (pattern not yet satisfied)
    assert all(i in allowed for i, p in enumerate(pieces) if p.isdigit())
    assert all(i not in allowed for i, p in enumerate(pieces) if not p.isdigit())
    assert eos not in allowed


def test_multi_char_token_spanning_transitions():
    dfa = compile_regex(r"\d{4}")
    pieces = ["20", "26", "202", "2026", "20267", "x"]
    eos = len(pieces)
    vocab = make_vocab(pieces, eos)
    allowed = allowed_token_ids(dfa, dfa.start, vocab)
    assert 0 in allowed  # "20" -> two digits, still live
    assert 3 in allowed  # "2026" -> exactly four digits, accepting is live
    assert 4 not in allowed  # "20267" -> five digits, dies
    assert 5 not in allowed  # "x"


def test_eos_only_when_accepting():
    dfa = compile_regex(r"\d{2}")
    pieces = list("0123456789")
    eos = len(pieces)
    vocab = make_vocab(pieces, eos)
    st = dfa.start
    assert eos not in allowed_token_ids(dfa, st, vocab)
    st = advance(dfa, st, "4")
    assert eos not in allowed_token_ids(dfa, st, vocab)  # one digit, not done
    st = advance(dfa, st, "2")
    assert eos in allowed_token_ids(dfa, st, vocab)  # two digits, may stop


def test_boundary_straddling_token():
    # A token that is quote+comma must only be legal where BOTH chars are legal.
    dfa = compile_regex(r'"\d",')  # a quoted single digit then a comma
    pieces = ['"', "5", '"', ",", '",', 'x']
    eos = len(pieces)
    vocab = make_vocab(pieces, eos)
    st = dfa.start
    st = advance(dfa, st, '"')
    st = advance(dfa, st, "5")
    # Now at the position before the closing quote. The token '",' (close-quote
    # + comma) should be legal here and complete the pattern.
    allowed = allowed_token_ids(dfa, st, vocab)
    assert 4 in allowed  # '",'
    end = consume(dfa, st, '",')
    assert dfa.is_accepting(end)


def test_token_that_overshoots_is_rejected():
    dfa = compile_regex(r"ab")
    pieces = ["a", "ab", "abc", "b"]
    eos = len(pieces)
    vocab = make_vocab(pieces, eos)
    allowed = allowed_token_ids(dfa, dfa.start, vocab)
    assert 0 in allowed and 1 in allowed  # "a", "ab"
    assert 2 not in allowed  # "abc" overshoots
    assert 3 not in allowed  # "b" wrong first char


def test_never_empty_for_reachable_state():
    dfa = compile_regex(r"(cat|dog)")
    pieces = list("catdog") + ["ca", "do"]
    eos = len(pieces)
    vocab = make_vocab(pieces, eos)
    assert len(allowed_token_ids(dfa, dfa.start, vocab)) > 0
