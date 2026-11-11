"""Constrained Decoding Engine.

A library that guarantees an LLM's output conforms to a schema or grammar by
masking invalid tokens at each decode step, so output is correct by
construction rather than correct-if-lucky.

Public surface for Phase 1 (regex / finite-automaton), Phase 2 (JSON schema via
a pushdown automaton), and Phase 3 (per-state token index, EBNF grammars,
streaming).
"""

from engine.automaton import NFA, DFA, State
from engine.regex_compiler import compile_regex
from engine.pushdown import PDA, Grammar, compile_grammar
from engine.json_schema import build_json_automaton, build_json_processor, schema_to_grammar
from engine.token_index import TokenIndex
from engine.ebnf import compile_ebnf, load_ebnf, build_ebnf_processor
from engine.streaming import StreamValidator, stream_constrained, stream_generate_hf

__all__ = [
    "NFA",
    "DFA",
    "State",
    "compile_regex",
    "PDA",
    "Grammar",
    "compile_grammar",
    "build_json_automaton",
    "build_json_processor",
    "schema_to_grammar",
    "TokenIndex",
    "compile_ebnf",
    "load_ebnf",
    "build_ebnf_processor",
    "StreamValidator",
    "stream_constrained",
    "stream_generate_hf",
]
