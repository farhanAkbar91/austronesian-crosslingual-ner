"""Lexical overlap analysis: target test entities vs. source training entities.

Measures how much of a target-language test set the source-language training
data has "seen" lexically:

* ``seen``    : exact match of the normalized surface AND entity type
                (or surface only when ``entity_type_aware=False``);
* ``similar`` : no exact match, but best character n-gram similarity against
                the source inventory is >= ``threshold``;
* ``unseen``  : neither.

This is a POST-TRAINING analysis. Target test annotations may be used here
but never for training or checkpoint selection (docs/methodology.md).
Counts are reported both per mention (every test occurrence) and per unique
(normalized surface, type) pair, plus per entity type.
"""

from __future__ import annotations

from collections import defaultdict

from axner.analysis.normalization import char_ngram_set, normalize_surface, similarity_fn
from axner.constants import ENTITY_TYPES
from axner.evaluation.entity_metrics import extract_spans

_EXAMPLE_CAP = 20


def extract_mentions(ds, label_names) -> list[dict]:
    """Extract entity mentions (type, span, surface) from a WikiANN-schema dataset."""
    mentions = []
    for row in ds:
        tags = [label_names[t] for t in row["ner_tags"]]
        tokens = row["tokens"]
        for stype, start, end in extract_spans(tags):
            mentions.append({
                "sentence_id": row.get("sentence_id", None),
                "type": stype,
                "start": start,
                "end": end,
                "surface": " ".join(tokens[start:end + 1]),
            })
    return mentions


def build_inventory(ds, label_names) -> dict:
    """Build a source-side entity inventory: type -> normalized surface -> count."""
    inventory: dict = defaultdict(lambda: defaultdict(int))
    for mention in extract_mentions(ds, label_names):
        inventory[mention["type"]][normalize_surface(mention["surface"])] += 1
    return {t: dict(surfaces) for t, surfaces in inventory.items()}


class _InventoryIndex:
    """Inverted index over inventory surfaces for fast best-similarity lookup."""

    def __init__(self, inventory: dict, n_range, similarity: str):
        self.similarity = similarity_fn(similarity)
        self.n_range = n_range
        self.surfaces: dict = {}          # type -> [normalized surfaces]
        self.gram_sets: dict = {}         # type -> [frozenset ngrams]
        self.postings: dict = {}          # type -> {ngram: [surface indices]}
        for etype, surfaces in inventory.items():
            self.surfaces[etype] = list(surfaces.keys())
            sets = [char_ngram_set(s, n_range) for s in surfaces.keys()]
            self.gram_sets[etype] = sets
            postings: defaultdict = defaultdict(list)
            for idx, grams in enumerate(sets):
                for gram in grams:
                    postings[gram].append(idx)
            self.postings[etype] = dict(postings)

    def best_similarity(self, surface: str, etype: str) -> float:
        target_grams = char_ngram_set(surface, self.n_range)
        candidates = set()
        for gram in target_grams:
            candidates.update(self.postings.get(etype, {}).get(gram, ()))
        best = 0.0
        sets = self.gram_sets.get(etype, [])
        for idx in candidates:
            best = max(best, self.similarity(target_grams, sets[idx]))
        return best


def classify_mentions(mentions, inventory, threshold: float = 0.5, n_range=(2, 3, 4),
                      entity_type_aware: bool = True, similarity: str = "dice",
                      max_examples: int = _EXAMPLE_CAP) -> dict:
    """Classify target-side mentions as seen / similar / unseen.

    ``inventory`` is the output of :func:`build_inventory` (source training
    entities). When ``entity_type_aware`` is False, exact matches ignore the
    entity type and similarity search spans all types.
    """
    index = _InventoryIndex(inventory, n_range, similarity)
    exact_keys: set = set()
    for etype, surfaces in inventory.items():
        for surface in surfaces:
            exact_keys.add((surface, etype) if entity_type_aware else surface)

    mention_counts = {"seen": 0, "similar": 0, "unseen": 0, "total": len(mentions)}
    unique_counts: defaultdict = defaultdict(int)
    examples = {"similar": [], "unseen": []}
    by_type: defaultdict = defaultdict(lambda: {"seen": 0, "similar": 0, "unseen": 0, "total": 0})

    for mention in mentions:
        normalized = normalize_surface(mention["surface"])
        etype = mention["type"]
        key = (normalized, etype) if entity_type_aware else normalized
        if key in exact_keys:
            category = "seen"
        else:
            search_type = etype if entity_type_aware else None
            if search_type is None:
                best = max(
                    (index.best_similarity(normalized, t) for t in ENTITY_TYPES if t in index.surfaces),
                    default=0.0,
                )
            else:
                best = index.best_similarity(normalized, search_type)
            category = "similar" if best >= threshold else "unseen"
            if category in examples and len(examples[category]) < max_examples:
                examples[category].append({
                    "surface": mention["surface"], "type": etype,
                    **({"best_similarity": round(best, 4)} if category == "similar" else {}),
                })
        mention_counts[category] += 1
        unique_counts[(normalized, etype)] += 1
        by_type[etype][category] += 1
        by_type[etype]["total"] += 1

    unique_seen = sum(
        1 for (surface, etype) in unique_counts
        if ((surface, etype) if entity_type_aware else surface) in exact_keys
    )
    return {
        "config": {
            "threshold": threshold,
            "n_range": list(n_range),
            "similarity": similarity,
            "entity_type_aware": entity_type_aware,
            "normalization": "unicode NFC + casefold + whitespace collapse",
            "matching": "exact normalized (surface, type) match; else best char-ngram similarity",
        },
        "mentions": mention_counts,
        "unique_surfaces": {
            "total": len(unique_counts),
            "seen": unique_seen,
            "similar_or_unseen": len(unique_counts) - unique_seen,
        },
        "by_type": {t: dict(v) for t, v in sorted(by_type.items())},
        "examples": examples,
    }


def overlap_report(target_test_ds, source_train_ds, label_names, threshold: float = 0.5,
                   n_range=(2, 3, 4), entity_type_aware: bool = True, similarity: str = "dice") -> dict:
    """End-to-end overlap report between a target test set and source training data."""
    inventory = build_inventory(source_train_ds, label_names)
    mentions = extract_mentions(target_test_ds, label_names)
    report = classify_mentions(
        mentions, inventory, threshold=threshold, n_range=n_range,
        entity_type_aware=entity_type_aware, similarity=similarity,
    )
    report["inventory"] = {
        "n_unique_entities": {t: len(s) for t, s in inventory.items()},
        "n_total_mentions": {t: sum(s.values()) for t, s in inventory.items()},
    }
    return report
