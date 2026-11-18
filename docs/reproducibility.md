# Reproducibility guide

How this repository makes runs traceable and repeatable — and where
determinism genuinely ends.

## Seeds and what they control

| quantity | seed | derivation |
|---|---|---|
| Dataset sampling (per language & split) | `data.sampling.seed` (42) | `derive_seed(base, "sample", lang, split)` |
| Multi-source composition shuffle | same base | `derive_seed(base, "compose", "shuffle")` |
| EXP-3 few-shot subsets | same base | `derive_seed(base, "fewshot", lang, max_size)` (nested prefixes) |
| Model init + training | `--seed` / `experiment.training_seeds` | passed to `set_deterministic` and Trainer (`seed`, `data_seed`) |
| Bootstrap resampling | `evaluation.bootstrap.seed` (42) | `derive_seed(seed, lang)` per target |

`derive_seed` (SHA-256 of `"<base>::<parts...>"` mod 2³¹−1) is stable across
Python/NumPy versions and independent of `PYTHONHASHSEED`.

Sampling uses `numpy.random.default_rng(seed).choice(n, k, replace=False)`
with sorted output. Stream compatibility of NumPy `Generator` is promised by
NEP 19; record the numpy version (it is captured in every run's
`config/environment.json`) when long-term bit-exact reproducibility matters.

`training.deterministic: true` seeds Python/NumPy/torch and forces cuDNN
deterministic mode. **Caveats:** GPU reductions can still be nondeterministic
in principle; dataloaders use 0 workers; identical results across different
torch versions or GPU models are NOT guaranteed. On the CPU-only development
machine this is a non-issue.

## Environment capture

Every run writes `config/environment.json`: Python version, platform, package
versions (torch, transformers, datasets, accelerate, numpy, pandas, pyarrow,
this package), CUDA availability, git commit + dirty flag (when run from a
clone).

## Artifact layout per run

```
outputs/<experiment_id>/<timestamp>-<label>-<uuid6>/
├── manifest.json                    # experiment, seed, variant tags, model id,
│                                    # dataset + revision, git commit, status, results
├── config/
│   ├── config_resolved.yaml         # full merged config snapshot
│   └── environment.json
├── data/
│   ├── sampling_indices.json        # per-source: language, split, n, seed, sha256, indices
│   ├── composition.json             # multi-source composition record (EXP-2)
│   ├── data_audit.json              # label schema, IOB2 scans, policies applied
│   ├── duplicates.json              # within- and cross-split duplicate report
│   ├── dataset_stats.json           # lengths, entity counts per split
│   └── truncation_stats.json        # how much data max_length cut
├── checkpoints/                     # HF checkpoints (git-ignored; optional in bundles)
├── logs/trainer_history.json
├── predictions/
│   ├── test_<lang>_sentence_level.jsonl   # word-level gold vs predicted tags
│   └── test_<lang>_entity_level.jsonl     # per-entity tp/fp/fn records
└── metrics/
    ├── validation_metrics.json            # training-time validation (selection metric)
    ├── validation_target_<lang>.json      # target validation (EXP-3 selection basis)
    ├── test_metrics_<lang>.json           # entity P/R/F1, per-type, error summary
    └── bootstrap_<lang>.json              # sentence-level bootstrap CI
```

A run is fully reconstructable from `manifest.json` + `config_resolved.yaml` +
`sampling_indices.json` given the same dataset revision.

## Moving artifacts between Colab/Kaggle and this repository

On the training machine (writes `outputs/<experiment_id>/<run_id>.bundle.zip`):

```
python scripts/export_artifacts.py --run-dir outputs/<experiment_id>/<run_id>
python scripts/export_artifacts.py --run-dir ... --include-checkpoints   # EXP-3 needs this
```

On this machine:

```
.venv\Scripts\python.exe scripts\import_artifacts.py --bundle bundles\<run_id>.bundle.zip
```

Import verifies the manifest and restores
`outputs/<experiment_id>/<run_id>/`; it refuses to overwrite without
`--overwrite`. Bundles without checkpoints carry everything needed for
analysis (notebook 04); EXP-3 checkpoint resolution requires
`--include-checkpoints` bundles.

## Validating before you trust a result

```
python scripts/validate_configs.py          # all configs validate
python -m pytest                            # 108 tests, CPU-only
python scripts/run_smoke_test.py            # offline end-to-end pipeline check
python scripts/run_experiment.py --config <exp> ... --dry-run   # data pipeline only
```

The smoke test validates the PIPELINE with a tiny random model; its numbers
are meaningless as research results.

## Analyzing results

- `scripts/aggregate_results.py` — aggregates test metrics across seeds per
  (experiment, variant, target); output includes an explicit note that this
  is run-level variation, not test-set uncertainty.
- `notebooks/04_results_analysis.ipynb` — per-run metrics, paired bootstrap
  between runs on the same test set, seen/unseen entity breakdowns.
- Confidence intervals: read `metrics/bootstrap_<lang>.json` per run. When
  comparing systems, prefer `paired_bootstrap` on the same sentences.

## Known nondeterminism boundaries

- GPU training (CUDA kernels, cuDNN autotune disabled but reductions remain).
- Transformers version differences (checkpoint format, Trainer internals).
- `datasets`/hub-side dataset updates — pin `data.dataset.revision` for
  frozen comparisons; audits in `data/audits/` record what was audited.
- NumPy `Generator` stream compatibility across major numpy changes.
