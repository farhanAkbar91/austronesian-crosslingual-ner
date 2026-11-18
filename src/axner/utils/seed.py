"""Deterministic seed handling.

Seed derivation uses SHA-256 so child seeds are stable across Python and
NumPy versions and independent of PYTHONHASHSEED. Full bitwise determinism of
GPU training is not guaranteed; see docs/reproducibility.md for the caveats.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np

_SEED_MODULUS = 2**31 - 1


def derive_seed(base_seed: int, *parts) -> int:
    """Deterministically derive a child seed from a base seed and labels.

    Examples:
        derive_seed(42, "sample", "id", "train")  -> per-language sampling seed
        derive_seed(42, "bootstrap")              -> bootstrap resampling seed
    """
    payload = "::".join([str(base_seed), *(str(p) for p in parts)]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % _SEED_MODULUS


def set_deterministic(seed: int) -> None:
    """Seed Python, NumPy, and (if installed) PyTorch RNGs.

    Also switches cuDNN to its deterministic mode when torch is available.
    GPU reductions can still be nondeterministic in principle; on CPU-only
    development machines this is a non-issue.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - GPU-only path
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
