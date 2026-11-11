"""Benchmark runner.

Modes
  baseline     prompt-only, unconstrained generation
  retry        prompt-and-retry: up to N attempts, stop at first schema-valid one
  constrained  generation with the engine's LogitsProcessor (valid by construction)

Grammars
  regex        Phase-1 finite-automaton tasks (REGEX_TASKS)
  json         JSON-schema tasks (JSON_SCHEMAS); constrained path needs Phase 2

Everything is device-agnostic (CPU locally; a free Colab/Kaggle T4 for latency).
The harness is frozen after Phase 0 and reused unchanged by later phases.

Usage:
  python -m bench.harness --mode baseline    --grammar json  --samples 5
  python -m bench.harness --mode constrained --grammar regex --samples 5
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import List, Optional

from bench import metrics
from bench.metrics import RunResult
from bench.schemas import JSON_SCHEMAS, REGEX_TASKS, JsonTask, RegexTask
from engine import config

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


# --------------------------------------------------------------------------- #
# Model loading / generation
# --------------------------------------------------------------------------- #
@dataclass
class Model:
    model: object
    tokenizer: object
    device: str
    model_id: str


def load_model(model_id: Optional[str] = None, device: Optional[str] = None) -> Model:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = model_id or config.DEFAULT_MODEL
    device = device or config.pick_device()
    tok = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.float32 if device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return Model(model=model, tokenizer=tok, device=device, model_id=model_id)


def _build_prompt(tok, system: str, instruction: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": instruction}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate(
    m: Model,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    seed: Optional[int],
    logits_processor=None,
):
    """Generate once. Returns (decoded_text, n_new_tokens, latency_seconds)."""
    import torch

    if seed is not None:
        torch.manual_seed(seed)
    enc = m.tokenizer(prompt, return_tensors="pt").to(m.device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=m.tokenizer.pad_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
    if logits_processor is not None:
        from transformers import LogitsProcessorList

        if hasattr(logits_processor, "reset"):
            logits_processor.reset()
        gen_kwargs["logits_processor"] = LogitsProcessorList([logits_processor])

    t0 = time.perf_counter()
    with torch.no_grad():
        out = m.model.generate(**enc, **gen_kwargs)
    latency = time.perf_counter() - t0

    in_len = enc["input_ids"].shape[1]
    new_ids = out[0][in_len:]
    text = m.tokenizer.decode(new_ids, skip_special_tokens=True)
    return text, int(new_ids.shape[0]), latency


# --------------------------------------------------------------------------- #
# Task runners
# --------------------------------------------------------------------------- #
_REGEX_SYSTEM = "You output only the requested value in the exact format asked, with no explanation, labels, or extra text."
_JSON_SYSTEM = "You output only a single JSON value matching the user's request. No prose, no markdown fences, no explanation."


def run_regex(m: Model, tasks: List[RegexTask], mode: str, samples: int, max_new_tokens: int) -> List[RunResult]:
    import re as _re

    from engine.regex_compiler import compile_regex
    from engine.token_bridge import TokenVocab
    from engine.masker import AutomatonLogitsProcessor

    vocab = TokenVocab.from_hf(m.tokenizer) if mode == "constrained" else None
    results: List[RunResult] = []
    for task in tasks:
        prompt = _build_prompt(m.tokenizer, _REGEX_SYSTEM, task.instruction)
        proc = None
        if mode == "constrained":
            proc = AutomatonLogitsProcessor(compile_regex(task.pattern), vocab)
        pat = _re.compile(task.pattern)
        for s in range(samples):
            text, n_new, lat = generate(
                m, prompt, max_new_tokens,
                do_sample=(samples > 1), temperature=0.7, seed=1000 + s,
                logits_processor=proc,
            )
            valid = pat.fullmatch(text.strip()) is not None
            results.append(RunResult(
                task=task.name, mode=mode, valid_json=valid, schema_valid=valid,
                latency_s=lat, new_tokens=n_new, attempts=1, output=text.strip(),
            ))
    return results


def run_json(m: Model, tasks: List[JsonTask], mode: str, samples: int, max_new_tokens: int, retries: int) -> List[RunResult]:
    if mode == "constrained":
        try:
            from engine.json_schema import build_json_processor  # Phase 2
            from engine.token_bridge import TokenVocab
        except ImportError:
            raise NotImplementedError(
                "constrained JSON needs Phase 2 (engine/json_schema.py). "
                "Baseline and retry modes work now; regex constrained works now."
            )
        vocab = TokenVocab.from_hf(m.tokenizer)

    results: List[RunResult] = []
    for task in tasks:
        prompt = _build_prompt(m.tokenizer, _JSON_SYSTEM, task.instruction)
        proc = None
        if mode == "constrained":
            proc = build_json_processor(task.schema, vocab)
        for s in range(samples):
            attempts = 1
            max_attempts = retries if mode == "retry" else 1
            while True:
                text, n_new, lat = generate(
                    m, prompt, max_new_tokens,
                    do_sample=(samples > 1 or mode == "retry"),
                    temperature=0.7, seed=1000 + s * 100 + attempts,
                    logits_processor=proc,
                )
                obj = metrics.parse_json(text)
                valid_json = obj is not None
                schema_valid = metrics.satisfies_schema(obj, task.schema)
                if schema_valid or attempts >= max_attempts:
                    break
                attempts += 1
            results.append(RunResult(
                task=task.name, mode=mode, valid_json=valid_json, schema_valid=schema_valid,
                latency_s=lat, new_tokens=n_new, attempts=attempts, output=text.strip(),
            ))
    return results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Constrained Decoding Engine benchmark")
    ap.add_argument("--mode", choices=["baseline", "retry", "constrained"], default="baseline")
    ap.add_argument("--grammar", choices=["regex", "json"], default="json")
    ap.add_argument("--model", default=None, help="HF model id (default: engine.config.DEFAULT_MODEL)")
    ap.add_argument("--samples", type=int, default=5, help="generations per task")
    ap.add_argument("--retries", type=int, default=3, help="max attempts in retry mode")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--out", default=None, help="CSV path (default: results/<grammar>_<mode>.csv)")
    ap.add_argument("--verbose", action="store_true", help="print each generation")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    m = load_model(args.model)
    print(f"[harness] model={m.model_id} device={m.device} mode={args.mode} grammar={args.grammar}")

    if args.grammar == "regex":
        mnt = args.max_new_tokens or 32
        results = run_regex(m, REGEX_TASKS, args.mode, args.samples, mnt)
    else:
        mnt = args.max_new_tokens or 160
        results = run_json(m, JSON_SCHEMAS, args.mode, args.samples, mnt, args.retries)

    if args.verbose:
        for r in results:
            flag = "OK " if r.schema_valid else "BAD"
            print(f"  [{flag}] {r.task:18s} {r.output[:70]!r}")

    aggs = metrics.aggregate(results)
    out_path = args.out or os.path.join(RESULTS_DIR, f"{args.grammar}_{args.mode}.csv")
    metrics.write_csv(out_path, aggs)

    overall_valid = sum(a.schema_valid_rate * a.n for a in aggs) / sum(a.n for a in aggs)
    print(f"\n[harness] {'task':18s} {'schema_valid':>12s} {'ms/tok':>9s} {'tok/s':>8s}")
    for a in aggs:
        print(f"          {a.task:18s} {a.schema_valid_rate:12.0%} {a.mean_ms_per_token:9.1f} {a.mean_tokens_per_s:8.1f}")
    print(f"\n[harness] overall schema-valid rate: {overall_valid:.1%}")
    print(f"[harness] wrote {out_path}")


if __name__ == "__main__":
    main()
