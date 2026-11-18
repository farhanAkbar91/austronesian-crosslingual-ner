# Data card — WikiANN for the Austronesian cross-lingual NER project

## Provenance

- **Dataset:** WikiANN (the PAN-X subset of Rahimi et al., 2019, *"Massive
  Multilingual Named Entity Recognition via Slug Alignment"*), distributed
  on the Hugging Face hub as [`wikiann`](https://huggingface.co/datasets/wikiann),
  one configuration per language.
- **Loaded via:** Hugging Face `datasets` (parquet backend);
  `axner.data.wikiann.load_wikiann`, cache in git-ignored `data/hf_cache`.
- **Audited:** 2026-11-18, `datasets` 4.8.5, via
  `scripts/audit_wikiann.py --languages jv su id ms tl en`; JSON reports in
  `data/audits/` (git-ignored; numbers recorded here).

## Features

Each row: `tokens` (list[str]), `ner_tags` (ClassLabel ids),
`langs` (list[str], per-token language), `spans` (list[str]).
This project uses `tokens` + `ner_tags`; `langs` is retained until alignment
and then dropped.

## Verified split sizes

| language | train | validation | test |
|---|---|---|---|
| id (Indonesian) | 20,000 | 10,000 | 10,000 |
| ms (Malay) | 20,000 | 1,000 | 1,000 |
| tl (Tagalog) | 10,000 | 1,000 | 1,000 |
| en (English) | 20,000 | 10,000 | 10,000 |
| **jv (Javanese)** | **100** | **100** | **100** |
| **su (Sundanese)** | **100** | **100** | **100** |

**Consequences for the research design (verified, not assumed):**

- EXP-0's "100 target-language training sentences" equals the **entire**
  jv/su train split — there is no held-back target training data beyond
  these 100 sentences.
- Target **test sets contain 100 sentences** each; evaluation is high-variance
  and bootstrap confidence intervals will be wide.
- EXP-3 few-shot sizes 10/50/100 are feasible exactly because train = 100.

Earlier draft documents that assumed larger jv/su splits should be updated to
these verified numbers.

## Label schema

Verified on all six configs: `O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC`
(ClassLabel order). The project remaps ids to its canonical order (identical
here) and verifies the label set on every load.

## IOB2 validity

Audit result: **0 invalid examples across all 18 (language, split) pairs**
(5,600 jv/su rows + 90,000 id/ms/tl/en rows scanned). Gold data is treated as
trustworthy with respect to IOB2 form; invalid-annotation policy remains
`report_only` as a safety net.

## Duplicates (exact token-sequence matching)

| language | within-split duplicate rows | cross-split shared sequences |
|---|---|---|
| jv | train 1, val 2, test 4 | train↔val 1, train↔test 1, val↔test 3 |
| su | train 4, val 1, test 2 | train↔val 5, train↔test 4, val↔test 1 |
| id | train 5,840, val 2,475, test 2,568 | train↔val 1,353, train↔test 1,341, val↔test 833 |
| ms | train 9,848, val 239, test 226 | train↔val 338, train↔test 353, val↔test 127 |
| tl | train 5,242, val 183, test 180 | train↔val 475, train↔test 488, val↔test 179 |
| en | train 366, val 110, test 113 | train↔val 257, train↔test 226, val↔test 139 |

Reading: WikiANN silver data contains substantial **within-split repetition**
(ms train: ~half of rows are repeats) and **cross-split leakage** (id/ms/tl
train share hundreds of sequences with test). The pipeline reports these
counts in every run (`data/duplicates.json`); the default policy is
`report_only`. `data.duplicates.policy: drop_train_dups` additionally removes
training rows whose token sequence appears in train/val/test — a *documented,
explicit* choice that changes the training distribution and should be reported
alongside results if used. Test splits are never altered.

Cross-lingual coincidences also exist (4–5 token sequences shared between id
train and jv/su test — mostly short punctuation/number sequences); these are
reported per run.

## Known quality caveats

- WikiANN is **silver** data (semi-supervised labels), not human-verified
  annotation; type errors and boundary noise are expected.
- `langs` marks per-token language; some sentences are mixed-language. This
  project treats a sentence's language as its config language and keeps all
  tokens.
- Small jv/su splits make the target evaluation noisy; treat fine-grained
  comparisons (e.g. between EXP-2 A/B) cautiously and lean on CIs.

## Licensing

The hub repository for `wikiann` does not state a clear license on its card
at audit time. WikiANN derives from Wikipedia text (per Rahimi et al., 2019),
which suggests source content under Creative Commons Attribution-ShareAlike
licensing, **but this has not been verified against the distribution's
terms**. Therefore:

- This repository **does not redistribute** WikiANN data or derived datasets
  (caches live in git-ignored `data/`).
- Before publishing any dataset artifact derived from WikiANN, **verify the
  applicable license terms** (hub card, Rahimi et al. 2019, and upstream
  Wikipedia licensing).

## Version pinning

`configs/data.yaml` exposes `data.dataset.revision` (null = latest at load
time). For frozen-data reruns, pin a hub revision and record it in every run
manifest (the runner stores `dataset_revision`).
