"""Model and tokenizer initialization.

torch/transformers are imported lazily so data/evaluation workflows run
without the training stack installed.
"""

from __future__ import annotations

from axner.errors import AxnerError
from axner.utils.logging import get_logger
from axner.utils.seed import set_deterministic

logger = get_logger("axner.models.init")


def resolve_device(device_pref: str = "auto") -> str:
    """Resolve the torch device: 'auto' picks cuda when available, else cpu."""
    if device_pref not in ("auto", "cpu", "cuda"):
        raise AxnerError(f"unknown device preference {device_pref!r}")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise AxnerError(
            "torch is not installed; install the training stack with "
            "`pip install -e \".[train]\"`"
        ) from exc
    if device_pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_pref == "cuda" and not torch.cuda.is_available():
        raise AxnerError("device='cuda' requested but CUDA is not available")
    return device_pref


def init_tokenizer(model_name: str, cache_dir: str | None = None):
    """Load a fast tokenizer; word-level alignment requires the fast API."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=True)
    if not hasattr(tokenizer, "word_ids"):
        raise AxnerError(
            f"tokenizer for {model_name!r} does not expose word_ids(); "
            "a fast tokenizer is required for subword alignment"
        )
    return tokenizer


def init_model(model_name: str, label_names, seed: int = 0,
               cache_dir: str | None = None, tiny_random: bool = False):
    """Initialize a token-classification model.

    ``tiny_random=True`` builds a tiny, randomly initialized model offline
    (pipeline validation only). Otherwise ``model_name`` is loaded from the
    Hugging Face hub/cache with a fresh classification head sized to
    ``label_names``.
    """
    set_deterministic(seed)
    if tiny_random:
        from axner.models.offline import build_tiny_model

        logger.warning("building TINY RANDOM model for pipeline validation (not research use)")
        return build_tiny_model(label_names, seed=seed)

    from transformers import AutoModelForTokenClassification

    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}
    return AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
        cache_dir=cache_dir,
    )
