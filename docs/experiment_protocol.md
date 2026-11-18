# Experiment protocol

Defines the planned experiment campaign (EXP-0 … EXP-3). **Nothing here has
been executed yet** — this is the run plan for Colab/Kaggle GPU sessions.
Every parameter lives in `configs/`; nothing is hardcoded.

## Conventions

- One **run** = one training configuration + one training seed.
- Training seeds: **42, 43, 44** (`experiment.training_seeds`).
- Sampling base seed: **42** (`experiment.sampling_seed`); per-language seeds
  are derived via SHA-256 (`axner.utils.seed.derive_seed`) — see
  `docs/reproducibility.md`.
- Evaluation always uses the **full target test split** (100 sentences for
  jv/su).
- Model: `FacebookAI/xlm-roberta-base`; hyperparameters from
  `configs/training.yaml` unless a config overrides them (lr 2e-5,
  batch 16, weight decay 0.01, warmup ratio 0.1).
- Epochs: **3** for 10k-sentence training runs; **10** for 100-sentence runs
  (EXP-0, EXP-3).
- fp16: enable on GPU via `--fp16`-style config override (`training.fp16`).

## Verified WikiANN sizes (inputs to this protocol)

| language | train | validation | test |
|---|---|---|---|
| id | 20,000 | 10,000 | 10,000 |
| ms | 20,000 | 1,000 | 1,000 |
| tl | 10,000 | 1,000 | 1,000 |
| en | 20,000 | 10,000 | 10,000 |
| jv | **100** | 100 | **100** |
| su | **100** | 100 | **100** |

`configs/project.yaml` records these (`known_max_train_sizes`,
`known_max_sizes_verified: true`); config validation rejects requested sizes
that exceed them.

## EXP-0 — In-language baseline (6 runs)

Train on 100 target-language training sentences (= the entire jv/su train
split); no source data. Validation on the full target validation split; test
on the target test split. 10 epochs.

- `--config exp0_inlanguage_baseline --target jv --seed {42,43,44}`
- `--config exp0_inlanguage_baseline --target su --seed {42,43,44}`

## EXP-1 — Single-source zero-shot transfer (4 sources × 3 seeds = 12 runs)

One source language at a time (10,000 train sentences, sampled without
replacement from the source train split); validation on 1,000 sampled source
validation sentences per split; **no target training data**. Each run
evaluates on BOTH targets (jv and su test) and additionally on both target
validation splits (for EXP-3 selection).

- `--config exp1_singlesource_zeroshot --source {id,ms,tl,en} --seed {42,43,44}`

## EXP-2 — Multi-source transfer (2 configs × 3 seeds = 12 runs)

Training data composed from per-language samples; total exactly 10,000
(enforced by config validation):

- **Config A** (`exp2a_multisource_id_ms`): 5,000 id + 5,000 ms.
- **Config B** (`exp2b_multisource_id_ms_en`): 3,333 id + 3,333 ms + 3,334 en.

Per-language composition samples use derived seeds; the concatenation is
shuffled with a derived seed. Validation: 1,000 sentences per source
validation split (concatenated). Each run evaluates on both targets.

- `--config exp2a_multisource_id_ms --seed {42,43,44}`
- `--config exp2b_multisource_id_ms_en --seed {42,43,44}`

## EXP-3 — Few-shot adaptation (≈ 2 arms × 3 sizes × 3 seeds = 18 runs)

Initialize from a source-trained checkpoint and fine-tune on 10 / 50 / 100
target sentences (jv).

- **Init selection (no target test data):** the checkpoint with the highest
  **target-validation** entity F1 among EXP-1 runs for the chosen source
  (default `id` in `exp3_fewshot_adaptation.yaml`), read from
  `metrics/validation_target_jv.json`. Set `init_from.checkpoint` to a path
  to override artifact-based resolution.
- **Nested subsets:** for each seed, the largest size (100) is sampled once
  and smaller sizes take prefixes — so size 10 ⊂ 50 ⊂ 100 within a seed.
- **Arms:** `--arm fewshot` (init from checkpoint) and `--arm target_only`
  (fresh XLM-R at the same sizes, baseline). EXP-0 provides the target-only
  baseline at size 100; the target_only arm covers 10 and 50.

```
--config exp3_fewshot_adaptation --arm fewshot     --fewshot-size {10,50,100} --seed {42,43,44}
--config exp3_fewshot_adaptation --arm target_only --fewshot-size {10,50,100} --seed {42,43,44}
```

Checkpoint artifacts must be available where EXP-3 runs: either run EXP-1 and
EXP-3 in the same environment, or export EXP-1 bundles **with checkpoints**
(`scripts/export_artifacts.py --include-checkpoints`) and import them first.

## Run summary

| experiment | runs |
|---|---|
| EXP-0 | 2 targets × 3 seeds = 6 |
| EXP-1 | 4 sources × 3 seeds = 12 (each evaluates jv + su) |
| EXP-2 | 2 compositions × 3 seeds = 6 |
| EXP-3 | 2 arms × 3 sizes × 3 seeds = 18 |
| **total** | **42 training runs** |

## Preflight before a cloud session

```
python scripts/validate_configs.py
python scripts/run_smoke_test.py                    # offline, tiny model
python scripts/run_experiment.py --config <exp> ... --dry-run   # data pipeline only
```

A dry run exercises loading, audits, sampling, composition, duplicate
checks, statistics, and artifact writing without downloading any model or
spending GPU time.

## Artifact naming

`outputs/<experiment_id>/<YYYYmmdd>-<HHMMSS>-<label>-<uuid6>/` with
`manifest.json` (experiment id, seed, variant tags, dataset revision, model
id, git commit), `config/config_resolved.yaml`, `data/` (sampling indices,
audits, duplicates, stats, truncation), `logs/trainer_history.json`,
`predictions/test_<lang>_{sentence,entity}_level.jsonl`, and
`metrics/` (`validation_metrics.json`, `validation_target_<lang>.json`,
`test_metrics_<lang>.json`, `bootstrap_<lang>.json`).

## Analysis after the campaign

1. Import bundles: `scripts/import_artifacts.py --bundle ...`
2. Aggregate across seeds: `scripts/aggregate_results.py`
3. Notebooks 04 for bootstrap comparisons and seen/unseen breakdowns.

## Status

- Protocol: **designed, configured, validated** (config validation passes).
- Executed: **nothing** — no results, no benchmark numbers, no claims.
