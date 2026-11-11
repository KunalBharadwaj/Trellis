"""Shared configuration constants.

The model id lives behind a single constant so it is trivially swappable
(the plan's requirement). Default is Qwen2.5-0.5B-Instruct: ungated, ~1GB,
runs on CPU. Override with the CDE_MODEL env var without touching code.
"""

import os

# Default open-source instruct model. Swap via `export CDE_MODEL=...`.
DEFAULT_MODEL = os.environ.get("CDE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

# Generation defaults used by the benchmark harness.
MAX_NEW_TOKENS = 256
# Greedy by default so the baseline vs constrained comparison is deterministic.
DO_SAMPLE = False
TEMPERATURE = 1.0


def pick_device():
    """Return 'cuda' when a GPU is present, else 'cpu'. Keeps code device-agnostic."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"
