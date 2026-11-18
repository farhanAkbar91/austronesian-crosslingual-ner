# Methodology

This document describes what the implemented code actually does. Statements
about experiments describe the *planned* design; **no experimental results
exist yet** and none are claimed anywhere in this repository.

## Task

Named Entity Recognition with three entity types — **PER, LOC, ORG** — over
IOB2-tagged tokens. The canonical label list (fixed id order used by every
module) is:

```
O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC
```

WikiANN label ids are remapped to this order at load time
(`axner.data.wikiann.load_wikiann`); a schema check runs on every loaded
split and mismatches raise `DataValidationError`.

## Data

WikiANN (Rahimi et al., 2019 — the PAN-X subset), loaded from the Hugging
Face hub dataset `wikiann` with one config per language. Verified split sizes
for the research languages are recorded in `docs/data_card.md`; the key
constraint is that **jv and su have exactly 100 examples per split**.

Silver-data caveats that the audit surfaces (report-only by default):
within-split duplicate token sequences (e.g. ~9.8k of 20k ms train rows),
cross-split duplicates (e.g. 1,341 shared sequences between id train and
test), and cross-lingual coincidences (4–5 token sequences shared between id
train and jv/su test). See `docs/data_card.md` for the full audit and
`configs/data.yaml` for the explicit policies (`report_only`,
`drop_train_dups`). **The code never silently repairs annotations.**

## Model and subword alignment

`FacebookAI/xlm-roberta-base` via `AutoModelForTokenClassification` with a
fresh head over the canonical labels. Inputs are built with
`AutoTokenizer` (fast tokenizer required; `word_ids` drives alignment).

Alignment policy (`axner.data.alignment`, `model.subword_policy:
first_subtoken_label`):

1. `is_split_into_words=True`, `truncation=True`, `max_length` from config.
2. The **first subtoken** of each word carries the word's IOB2 tag id.
3. Special tokens and all later subtokens of a word get `-100`
   (`IGNORE_INDEX`) and are excluded from the loss, from validation metrics,
   and from evaluation.
4. Words cut by truncation are excluded from evaluation; the truncation rate
   is written to `data/truncation_stats.json` in every run. A word whose
   first subtoken survived truncation keeps its label even if later subtokens
   were cut (documented boundary behavior).

## Evaluation

Entity-level, **exact span matching**: a predicted entity is correct only
when its type AND both token boundaries match a gold entity
(`axner.evaluation.entity_metrics`).

- Micro precision / recall / F1 over all sentences and types.
- Per-type P/R/F1 for PER, LOC, ORG.
- Error taxonomy: type confusions (same boundaries, wrong type; counted in a
  gold→pred confusion matrix), boundary errors (same type, overlapping but
  not exact), spurious, missed.

Gold sequences are validated IOB2 upstream. **Model predictions** may
violate IOB2; the documented decode-time recovery treats an `I-X` that does
not continue an open `X` span as starting a new span. This affects only
predicted sequences, never gold annotations.

Evaluation runs on the **full target test split** (which is 100 sentences for
jv/su). Test splits are never altered by any policy.

## Statistical protocol

Two distinct sources of uncertainty, deliberately separated:

1. **Seed variation** (`axner.evaluation.aggregation.aggregate_seeds`):
   mean/±std/min/max of metrics across training runs with different seeds of
   the same configuration. This quantifies *run-level* variation.
   **Repeated seeds are not independent test datasets** and are never pooled
   as if they were.
2. **Test-set sampling uncertainty**
   (`axner.evaluation.bootstrap.bootstrap_ci`): percentile bootstrap over
   test sentences (resampled with replacement, default 1000 replicates,
   seed 42) for one system's entity micro-F1. With 100-sentence test sets,
   expect wide intervals.
3. **System comparison** (`paired_bootstrap`): both systems resampled on the
   SAME sentence indices per replicate; reports the CI of the F1 difference
   and an approximate two-sided bootstrap p-value. Valid only for runs that
   evaluated the same test sentences.

## Lexical overlap analysis

`axner.analysis.lexical_overlap` compares target **test** entity mentions
against the source **training** entity inventory (post-training analysis
only — target test annotations are never used for training or checkpoint
selection).

- **Normalization (fixed):** Unicode NFC → casefold → strip → collapse
  internal whitespace to single spaces.
- **Exact match:** normalized surface + entity type (or surface only when
  `entity_type_aware=False`).
- **Similarity:** Dice coefficient over character 2–4-grams of the normalized
  surface wrapped in `^...$` boundary markers; `jaccard` also available.
  Configurable threshold (default 0.5). An inverted index keeps the search
  fast.
- **Categories:** `seen` (exact match), `similar` (best similarity ≥
  threshold), `unseen`. Counts are reported per mention, per unique
  (surface, type) pair, and per entity type, with capped examples.

Joining overlap categories with per-entity prediction outcomes
(`predictions/test_<lang>_entity_level.jsonl`) yields seen/unseen F1 breakdowns
(notebook 04).

## Checkpoint selection and leakage policy

- During training, the best epoch is selected by **validation** entity F1
  (`training.checkpoint.selection_metric`), never test data.
- EXP-1/EXP-2 runs additionally evaluate each target-language **validation**
  split (`evaluation.eval_target_validation`), producing
  `metrics/validation_target_<lang>.json`.
- EXP-3 initial checkpoints are resolved by the highest
  **target-validation** entity F1 (`init_from.selection_metric:
  target_validation_entity_f1`). Target **test** data is never consulted.

## Not yet done

- No full training campaign has been executed; every metric in `outputs/`
  comes from smoke tests with a tiny random model or is absent.
- GPU training behavior (fp16, throughput, determinism on CUDA) is unverified.
- Colab/Kaggle instructions are written but unvalidated end-to-end.
