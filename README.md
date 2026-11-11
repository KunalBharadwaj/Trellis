# Constrained Decoding Engine

**Make a language model's output valid by construction.** This library compiles a
schema or grammar into a state machine and, at every decoding step, blocks every
token that would break the structure. The model literally cannot produce invalid
output — not "usually valid," not "valid after a retry," but valid every single
time.

It drops into an existing Hugging Face `generate()` call as a `LogitsProcessor`,
so you get the guarantee without touching the rest of your pipeline.

On a tiny 0.5B model, this takes JSON output from **37% valid to 100%** — and on
the nested-object and array cases, where plain prompting scores a flat **0%**, it
still hits **100%**. A per-state token cache keeps it fast enough that the
model's forward pass, not the constraint, is the bottleneck.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from engine.token_bridge import TokenVocab
from engine.json_schema import build_json_processor

tok   = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

schema = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
    "additionalProperties": False,
}

vocab = TokenVocab.from_hf(tok)
proc  = build_json_processor(schema, vocab)

ids = tok("Return JSON for a person.", return_tensors="pt").input_ids
out = model.generate(ids, max_new_tokens=160,
                     logits_processor=LogitsProcessorList([proc]))
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))  # always parses
```

## The problem it kills

Structured output is where a lot of LLM features quietly break: tool calls, data
extraction, anything that pipes JSON into the next step. Prompting helps but
guarantees nothing, and smaller models fail constantly.

I measured exactly how badly on Qwen2.5-0.5B. Asked politely for JSON, it produces
**nothing valid** for arrays or nested objects, and retrying the same prompt
doesn't rescue it — it just burns more calls for the same wrong answer.
Constrained decoding ends the problem outright, because an invalid token is never
even on the table.

## Results

Qwen2.5-0.5B-Instruct, 5 samples per task, on a free Colab T4. Raw CSVs in
[`results/`](results/) — reproduced to the decimal across two runs.

JSON schema — share of outputs that satisfy the schema:

| Task | Prompt only | Prompt + retry (≤3) | Constrained |
|------|:-----------:|:-------------------:|:-----------:|
| flat object        | 100% | 100% | **100%** |
| enum field         | 100% | 100% | **100%** |
| optional field     |  20% |  60% | **100%** |
| array of strings   |   0% |   0% | **100%** |
| nested object      |   0% |   0% | **100%** |
| array of objects   |   0% |   0% | **100%** |
| **overall**        | **37%** | **43%** | **100%** |

Regex / fixed formats — share valid:

| Task | Prompt only | Constrained |
|------|:-----------:|:-----------:|
| ISO date `YYYY-MM-DD` |   0% | **100%** |
| hex color `#RRGGBB`   |  20% | **100%** |
| US phone              |  60% | **100%** |
| colour enum, semver, yes/no | 100% | **100%** |
| **overall**           | **63%** | **100%** |

The retry column is the quiet punchline: three attempts move overall validity
from 37% to just 43%, because the model fails the same schemas the same way every
time. Retrying a broken approach is not a strategy. Constraining the decoder is.

### Speed

Masking isn't free — the obvious implementation re-scans the whole ~150k-token
vocabulary on every step, roughly tripling per-token time (about 30 → 210
ms/token for JSON). But the set of legal tokens only depends on the current
state, so it can be cached per state and each step becomes a lookup. With that
index, constrained decoding went from **174 to 31 ms/token — a 5.7× speedup, with
byte-for-byte identical output.** At that point the mask has effectively
disappeared from the budget; you're paying for the model, not the guarantee. See
[`results/phase3_speed.csv`](results/phase3_speed.csv).

## How it works

Four moving parts:

1. **Compile the schema into an automaton.** A regex becomes a DFA, built from
   scratch via Thompson construction (`engine/automaton.py`) — no `re` module
   doing the heavy lifting. A JSON schema becomes a *pushdown* automaton, a state
   machine with a stack, so it can track nested braces and brackets
   (`engine/pushdown.py`, `engine/json_schema.py`).

2. **Map automaton states to allowed tokens.** This is the part that makes or
   breaks the whole thing. The automaton thinks in characters; the model emits
   tokens, which are usually several characters and can straddle a boundary — a
   single token whose text is `",` is a closing quote *and* a comma. For the
   current state, `engine/token_bridge.py` computes exactly which token IDs keep
   the output on a valid path. Most of the engineering time lives here.

3. **Mask and sample.** `engine/masker.py` sets every illegal token's logit to
   −∞ so sampling can only land on a legal one, then advances the automaton by
   whatever was chosen. It never masks *every* token (that would deadlock
   generation) and only unlocks the end-of-sequence token once the output is
   genuinely complete.

4. **Precompute for speed.** `engine/token_index.py` caches step 2 per state, so
   the expensive scan runs once per state instead of once per token.

A detail I'm fond of: nesting depth is tracked by the grammar's *structure*, not
by counting brace characters. So a `{` sitting inside a string value is just
text — it never touches the stack, and the "token that straddles a brace" case
that trips up naive approaches simply can't happen here.

## Project layout

```
engine/
  automaton.py       regex → NFA → DFA (Thompson construction)
  regex_compiler.py  regex string → DFA
  token_bridge.py    map automaton states to allowed token IDs
  masker.py          the LogitsProcessor
  pushdown.py        stack-based automaton for nested grammars
  json_schema.py     JSON Schema → pushdown grammar
  token_index.py     per-state token cache (the speed fix)
  ebnf.py            EBNF grammars → pushdown (supports recursion)
  streaming.py       token-by-token streaming + stream validation
bench/               benchmark harness, metrics, test schemas
grammars/            example grammars (tool_call.ebnf)
results/             benchmark CSVs
tests/               59 tests, none need a model
notebooks/           self-contained Colab/Kaggle benchmark
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                       # 59 tests, no model or GPU needed
```

The correctness story stands entirely on the tests — the automata, the token
bridge, the edge cases, and an invariant check that the masker never deadlocks
and that every point where it allows a stop is already valid. You don't need to
load a model to trust the engine.

For the benchmark numbers you do need a model. Correctness runs fine on CPU; a
GPU only matters for the latency figures. The easiest route is
[`notebooks/benchmark_colab.ipynb`](notebooks/benchmark_colab.ipynb): open it in
Colab or Kaggle, pick a T4 runtime, run all. It embeds the engine source, so
there's nothing to clone or upload, and it also runs three live demos — the index
speedup, a recursive tool-call grammar, and streaming.

Swap models with the `CDE_MODEL` environment variable or `engine/config.py`.

## Beyond JSON: recursive grammars

The same pushdown machine runs arbitrary EBNF grammars, including recursive ones.
`grammars/tool_call.ebnf` defines a tool-call DSL where an argument can itself be
a call — `add(mul(2, 3), neg(-4))` — nesting to any depth. A plain DFA cannot
represent that; the stack can, and you can watch it grow (nesting depth 5 → 11 →
17 as the calls nest deeper). Every generation under it is a valid call by
construction.

## What's in and what's out

In: regular expressions (a practical subset), JSON schemas with objects, arrays,
nesting, scalar types, enums, and required/optional fields, recursive EBNF
grammars, and streaming with early validation.

Out: no training or fine-tuning — this is entirely inference-time; recursive
`$ref` in JSON schemas (use an EBNF grammar instead); and it isn't trying to
out-run production engines like Outlines or XGrammar.

## Two things worth taking away

- **The token boundary is the whole game.** Nearly all the debugging went into
  tokens that cross a structural boundary and into never letting the mask block
  every option. Get that right and the rest follows.
- **Correctness and speed were the same idea twice.** The per-state token index
  that makes it fast is just a cached copy of the per-state legality check that
  makes it correct — same computation, memoized.

## One caveat about constrained decoding in general

It needs the logits at every step, which means a local model or a server you
control — a hosted chat API won't do, since it only hands back finished text.
Outlines, guidance, and XGrammar are the references for the production-grade
version of these ideas.
