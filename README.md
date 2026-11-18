# austronesian-crosslingual-ner

Research infrastructure for an undergraduate project on **cross-lingual Named
Entity Recognition transfer** from Austronesian and non-Austronesian source
languages to low-resource target languages (primarily **Javanese**, with
**Sundanese** for replication), using **WikiANN** silver data and
**XLM-RoBERTa-base**.

This repository contains the *infrastructure*: data engineering, training and
evaluation code, configuration, artifact management, tests, and
documentation. **No model training has been executed and no experimental
results are claimed.** Full training runs are intended for Google Colab or
Kaggle GPU environments.

## Status

| Component | Status |
|---|---|
| Config system (YAML + fail-early validation) | Implemented, tested |
| WikiANN loading, metadata, split-size audit | Implemented, verified on all 6 research languages |
| IOB2 / label-schema validation (no silent repair) | Implemented, tested |
| Subword alignment (first-subtoken policy) | Implemented, tested |
| Reproducible sampling + multi-source composition | Implemented, tested |
| Duplicate detection & split-isolation reporting | Implemented, tested |
| Training orchestration (HF Trainer, XLM-R) | Implemented; **CPU smoke-tested only** (tiny model) — full XLM-R training unverified |
| Entity-level metrics (exact span matching) | Implemented, tested |
| Sentence-level bootstrap + paired bootstrap | Implemented, tested |
| Seed-level aggregation | Implemented, tested |
| Prediction export (sentence/entity JSONL) | Implemented, tested |
| Lexical overlap analysis | Implemented, tested |
| Artifact runs, manifests, bundles (import/export) | Implemented, tested |
| Offline CPU smoke test (synthetic + tiny model) | Implemented, passing locally |
| Colab/Kaggle instructions | Written, **not yet validated** |
| EXP-0..3 experiments | Configured; **not executed** |
| Notebooks 01–04 | Implemented; light runtime validation pending |

## Research scope

- **Sources:** Indonesian (`id`), Malay (`ms`), Tagalog (`tl`), English (`en`)
- **Targets:** Javanese (`jv`), Sundanese (`su`)
- **Task:** NER, entity types PER/LOC/ORG, IOB2 labels
- **Model:** `FacebookAI/xlm-roberta-base` (PyTorch + Hugging Face Transformers)
- **Data:** WikiANN (Hugging Face `wikiann`, one config per language)

**Important verified data fact:** the current WikiANN hub dataset gives `jv`
and `su` **exactly 100 examples per split**. EXP-0's "100 target sentences"
is therefore the *entire* train split, and target test sets contain 100
sentences (wide bootstrap CIs expected). See `docs/data_card.md`.

## Repository layout

```
configs/
  project.yaml data.yaml model.yaml training.yaml evaluation.yaml   # base configs
  experiments/exp0..exp3, smoke_test.yaml                            # per-experiment configs
docs/            methodology, protocol, data card, reproducibility
notebooks/       01 audit, 02 exploration, 03 pilot, 04 results analysis
scripts/         CLI entry points (see below)
src/axner/       config/ data/ models/ evaluation/ analysis/ artifacts/ experiments/ utils/
tests/           pytest suite (synthetic fixtures; no downloads, no GPU)
data/            git-ignored: HF cache, audits, downloads
outputs/         git-ignored: experiment run directories
```

## Local setup (Windows 10, CPU-only)

Requires Python **3.10 or 3.11** (developed on 3.11). From the repository root
in PowerShell or Git Bash:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip

# CPU-only torch first (smaller download than the default CUDA wheel):
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Project + training stack + dev tools:
.venv\Scripts\python.exe -m pip install -e ".[train,dev]"
```

Base dependencies alone (`pip install -e .`) suffice for data analysis and the
full test suite; `train` adds torch/transformers/accelerate for training and
smoke tests; `dev` adds pytest and matplotlib.

## Common commands

```powershell
# Validate every configuration (fast, no downloads):
.venv\Scripts\python.exe scripts\validate_configs.py

# Run the test suite (synthetic fixtures, CPU, no downloads):
.venv\Scripts\python.exe -m pytest

# Offline pipeline smoke test (synthetic data + tiny random model):
.venv\Scripts\python.exe scripts\run_smoke_test.py

# Smoke test on a tiny real WikiANN slice (downloads only the jv config):
.venv\Scripts\python.exe scripts\run_smoke_test.py --real-data

# Audit WikiANN languages (metadata, IOB2, duplicates; downloads per language):
.venv\Scripts\python.exe scripts\audit_wikiann.py --languages jv su id ms tl en

# Dry-run an experiment (data pipeline + artifacts only, no model):
.venv\Scripts\python.exe scripts\run_experiment.py --config exp0_inlanguage_baseline --target jv --dry-run

# A real training run (GPU recommended; see Colab/Kaggle sections):
.venv\Scripts\python.exe scripts\run_experiment.py --config exp1_singlesource_zeroshot --source id --seed 42

# Aggregate results across seeds:
.venv\Scripts\python.exe scripts\aggregate_results.py
```

Experiment configs also load by bare id: `--config exp1_singlesource_zeroshot`.

## Running one experiment (run matrix)

Each run = one training configuration + one seed. Campaign loops live in
`docs/experiment_protocol.md`.

| Experiment | Command sketch |
|---|---|
| EXP-0 in-language baseline | `run_experiment.py --config exp0_inlanguage_baseline --target jv --seed 42` |
| EXP-1 zero-shot (per source) | `run_experiment.py --config exp1_singlesource_zeroshot --source id --seed 42` |
| EXP-2 multi-source A | `run_experiment.py --config exp2a_multisource_id_ms --seed 42` |
| EXP-2 multi-source B | `run_experiment.py --config exp2b_multisource_id_ms_en --seed 42` |
| EXP-3 few-shot | `run_experiment.py --config exp3_fewshot_adaptation --arm fewshot --fewshot-size 50 --seed 42` |

Each run writes a fully traceable directory under `outputs/<experiment_id>/`
(config snapshot, environment, sampling indices, audits, checkpoints, logs,
predictions, metrics, manifest).

## Google Colab setup (instructions — not yet validated)

1. Runtime → change runtime type → **GPU (T4)**.
2. Get the code onto the VM (either):
   - `!git clone <your-repo-url> && %cd austronesian-crosslingual-ner`, or
   - upload a zip of the repository and unzip it.
3. Install (Colab ships torch; install the rest):

   ```
   !pip install -e ".[train]"
   ```
4. Preflight without spending GPU time:

   ```
   !python scripts/validate_configs.py
   !python scripts/run_experiment.py --config exp0_inlanguage_baseline --target jv --dry-run
   ```
5. Train (persistent storage recommended — mount Drive and use it for
   `--output-base` so artifacts survive session resets):

   ```
   from google.colab import drive
   drive.mount('/content/drive')
   !python scripts/run_experiment.py --config exp1_singlesource_zeroshot --source id --seed 42 \
       --output-base /content/drive/MyDrive/axner_outputs --fp16
   ```
   (`--fp16` is honored only on CUDA; `configs/*.yaml` can also set
   `training.fp16`.)
6. Export a bundle and download it:

   ```
   !python scripts/export_artifacts.py --run-dir <path printed by the run>
   ```
   then download `outputs/<experiment_id>/<run_id>.bundle.zip` from the file
   browser (or copy it from Drive).

## Kaggle setup (instructions — not yet validated)

1. New Notebook → Settings → Accelerator → **GPU**; Internet **on**.
2. Add your repository as a Kaggle Dataset (zip upload), then in the notebook:

   ```
   !unzip -q /kaggle/input/<dataset-name>/austronesian-crosslingual-ner.zip -d /kaggle/working
   %cd /kaggle/working/austronesian-crosslingual-ner
   ```
3. `!pip install -e ".[train]"` (Kaggle images include torch; check versions).
4. Run the same commands as above, writing outputs under `/kaggle/working`
   (persisted as notebook output), e.g. `--output-base /kaggle/working/outputs`.
5. Export bundles and download them from the notebook **Output** tab.

## Artifact transfer (Colab/Kaggle → local)

```powershell
# On your machine, restore a bundle into outputs/:
.venv\Scripts\python.exe scripts\import_artifacts.py --bundle bundles\<run_id>.bundle.zip

# EXP-3 needs checkpoint weights on the analysis machine:
#   export with --include-checkpoints before transferring
```

Bundles are zips of one run directory (manifest verified on import). Without
checkpoints they contain everything needed for notebooks 01/02/04: metrics,
predictions, bootstrap files, sampling indices, audits.

## Notebooks

Notebooks call `src/axner` functions; no core logic is duplicated.

1. `01_dataset_audit.ipynb` — metadata, split sizes, IOB2, duplicates
2. `02_data_exploration.ipynb` — statistics, examples, lexical overlap preview
3. `03_pilot_experiment.ipynb` — dry-run + offline smoke (pipeline validation)
4. `04_results_analysis.ipynb` — aggregation, bootstrap, seen/unseen analysis

## Documentation

- `docs/methodology.md` — task, alignment policy, metrics, statistics protocol
- `docs/experiment_protocol.md` — EXP-0..3 run matrices and selection rules
- `docs/data_card.md` — WikiANN provenance, verified sizes, licensing caveats
- `docs/reproducibility.md` — seeds, artifacts, bundles, determinism limits

## Data & licensing

WikiANN is distributed via the Hugging Face hub (`wikiann`). This repository
does **not** distribute WikiANN data or derived datasets; caches live in
git-ignored `data/`. **Verify the dataset's license terms before
redistributing any data or derived artifacts** — see `docs/data_card.md`.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest            # everything CPU-only
.venv\Scripts\python.exe -m pytest -m "not torch"   # skip model-stack tests
```

108 tests (as of the current commit) covering IOB2 validation, alignment and
subword handling, reproducible sampling, composition, split isolation,
config validation, entity metrics, bootstrap, prediction serialization,
artifact/bundle round trips, lexical overlap, and two torch-marked
end-to-end pipeline tests using the offline tiny model.
