"""Metric helpers shared by every phase: does the output parse, does it satisfy
the schema, and latency aggregation. Pure stdlib + jsonschema (no torch), so it
is unit-testable on its own.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import jsonschema

    _HAVE_JSONSCHEMA = True
except Exception:  # pragma: no cover
    _HAVE_JSONSCHEMA = False


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_text(text: str) -> Optional[str]:
    """Best-effort extraction of a JSON value from model output.

    Tries, in order: a ```json fenced block, the first balanced {...} or [...]
    span, then the stripped whole string. This is intentionally generous — it is
    the *baseline's* best shot at parsing, so the prompt-only number is not
    unfairly low."""
    if text is None:
        return None
    m = _FENCE.search(text)
    if m:
        inner = m.group(1).strip()
        if inner:
            return inner
    span = _first_balanced_span(text)
    if span is not None:
        return span
    stripped = text.strip()
    return stripped or None


def _first_balanced_span(text: str) -> Optional[str]:
    """Return the first balanced {...}/[...] substring, respecting strings."""
    opens = {"{": "}", "[": "]"}
    for start, ch in enumerate(text):
        if ch in opens:
            close = opens[ch]
            depth = 0
            in_str = False
            esc = False
            for j in range(start, len(text)):
                c = text[j]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                    continue
                if c == '"':
                    in_str = True
                elif c == ch:
                    depth += 1
                elif c == close:
                    depth -= 1
                    if depth == 0:
                        return text[start : j + 1]
            return None  # opened but never closed
    return None


def parse_json(text: str):
    """Parse model output into a Python object, or None if it does not parse."""
    candidate = extract_json_text(text)
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None


def is_valid_json(text: str) -> bool:
    return parse_json(text) is not None


def satisfies_schema(obj, schema: dict) -> bool:
    """True if ``obj`` validates against ``schema``. Requires jsonschema."""
    if obj is None:
        return False
    if not _HAVE_JSONSCHEMA:  # pragma: no cover
        raise RuntimeError("jsonschema not installed; cannot validate schema")
    try:
        jsonschema.validate(obj, schema)
        return True
    except jsonschema.ValidationError:
        return False
    except jsonschema.SchemaError as e:  # pragma: no cover
        raise RuntimeError(f"bad schema: {e}") from e


@dataclass
class RunResult:
    """One generation attempt's outcome."""

    task: str
    mode: str
    valid_json: bool
    schema_valid: bool
    latency_s: float
    new_tokens: int
    attempts: int = 1
    output: str = ""

    @property
    def ms_per_token(self) -> float:
        return (self.latency_s * 1000.0 / self.new_tokens) if self.new_tokens else 0.0


@dataclass
class Aggregate:
    """Rolled-up metrics for a (task, mode) group."""

    task: str
    mode: str
    n: int
    valid_json_rate: float
    schema_valid_rate: float
    mean_latency_s: float
    mean_ms_per_token: float
    mean_tokens_per_s: float
    mean_attempts: float


def aggregate(results: List[RunResult]) -> List[Aggregate]:
    """Group results by (task, mode) and compute rates + latency."""
    groups: dict = {}
    for r in results:
        groups.setdefault((r.task, r.mode), []).append(r)
    out: List[Aggregate] = []
    for (task, mode), rs in sorted(groups.items()):
        n = len(rs)
        tok_per_s = [
            (r.new_tokens / r.latency_s) for r in rs if r.latency_s > 0 and r.new_tokens
        ]
        out.append(
            Aggregate(
                task=task,
                mode=mode,
                n=n,
                valid_json_rate=sum(r.valid_json for r in rs) / n,
                schema_valid_rate=sum(r.schema_valid for r in rs) / n,
                mean_latency_s=statistics.fmean(r.latency_s for r in rs),
                mean_ms_per_token=statistics.fmean(r.ms_per_token for r in rs if r.new_tokens)
                if any(r.new_tokens for r in rs)
                else 0.0,
                mean_tokens_per_s=statistics.fmean(tok_per_s) if tok_per_s else 0.0,
                mean_attempts=statistics.fmean(r.attempts for r in rs),
            )
        )
    return out


# CSV field order for results/*.csv (stdlib csv, no pandas dependency required).
CSV_FIELDS = [
    "task",
    "mode",
    "n",
    "valid_json_rate",
    "schema_valid_rate",
    "mean_latency_s",
    "mean_ms_per_token",
    "mean_tokens_per_s",
    "mean_attempts",
]


def write_csv(path: str, aggs: List[Aggregate]) -> None:
    import csv

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for a in aggs:
            w.writerow({k: getattr(a, k) for k in CSV_FIELDS})
