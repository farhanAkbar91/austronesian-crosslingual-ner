"""Surface-form normalization and character n-gram similarity.

Normalization rules (fixed, documented in docs/methodology.md):
  1. Unicode NFC normalization;
  2. casefold (aggressive lowercasing, e.g. German sharp-s handling);
  3. strip and collapse every internal whitespace run to a single space.

Similarity: Dice coefficient over character n-grams of the normalized string
wrapped in boundary markers (``^...$``), n in a configurable range
(default 2-4).
"""

from __future__ import annotations

import unicodedata

from axner.constants import ENTITY_TYPES


def normalize_surface(surface: str) -> str:
    """Apply the documented normalization pipeline to one surface form."""
    surface = unicodedata.normalize("NFC", surface)
    surface = surface.casefold().strip()
    return " ".join(surface.split())


def char_ngram_set(surface: str, n_range=(2, 3, 4)) -> frozenset[str]:
    """Character n-gram set of a normalized surface with boundary markers."""
    marked = f"^{surface}$"
    grams = set()
    for n in n_range:
        if n <= len(marked):
            grams.update(marked[i:i + n] for i in range(len(marked) - n + 1))
    return frozenset(grams)


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def dice(a: frozenset, b: frozenset) -> float:
    total = len(a) + len(b)
    if total == 0:
        return 0.0
    return 2 * len(a & b) / total


def similarity_fn(name: str):
    """Resolve a similarity function by name ('dice' or 'jaccard')."""
    if name == "dice":
        return dice
    if name == "jaccard":
        return jaccard
    raise ValueError(f"unknown similarity {name!r} (expected 'dice' or 'jaccard')")
