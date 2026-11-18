"""Experiment runner: config -> data -> alignment -> training -> evaluation -> artifacts.

``execute_run`` executes ONE training/evaluation run (one seed, one training
language configuration). Campaign loops (seed lists, source matrices) live in
scripts and notebooks, which call ``execute_run`` repeatedly.

EXP-3 checkpoint selection uses TARGET VALIDATION metrics only
(``metrics/validation_target_<lang>.json``); target test data is never used
for training or checkpoint selection.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from axner.artifacts.environment import capture_environment_info
from axner.artifacts.manager import (
    artifact_paths,
    create_run_directory,
    load_json,
    save_json,
    save_yaml,
    update_manifest,
    write_manifest,
)
from axner.config.loader import load_experiment_config
from axner.config.validation import validate_or_raise
from axner.constants import (
    LABEL_NAMES,
    SOURCE_LANGUAGES,
    TARGET_LANGUAGES,
)
from axner.data.alignment import align_dataset, truncation_stats
from axner.data.sampling import compose_multisource, language_sampling_seed, sample_split
from axner.data.stats import dataset_statistics, duplicates_report
from axner.data.validation import apply_invalid_policy, check_label_schema, scan_iob2_issues
from axner.evaluation.bootstrap import bootstrap_ci
from axner.evaluation.entity_metrics import compute_entity_metrics, error_summary, per_type_metrics
from axner.evaluation.export import build_sentence_records, export_jsonl, build_entity_records
from axner.errors import ArtifactError, ConfigError, DataValidationError
from axner.utils.logging import get_logger
from axner.utils.seed import derive_seed, set_deterministic

logger = get_logger("axner.experiments.runner")


def load_run_config(path_or_name) -> dict:
    """Load and validate an experiment config by path or experiment id."""
    cfg = load_experiment_config(path_or_name)
    return validate_or_raise(cfg)


# --------------------------------------------------------------------------
# run-point resolution (which source/target/arm/size this specific run uses)
# --------------------------------------------------------------------------

def _test_language_mode(cfg: dict) -> str:
    """Experiment-level override wins over the evaluation default."""
    return cfg["experiment"].get("test_language_mode") or cfg["evaluation"]["test_language_mode"]


def resolve_run_point(cfg: dict, seed=None, target=None, source=None, arm=None,
                      fewshot_size=None) -> dict:
    """Resolve the concrete run point from the experiment specification."""
    experiment = cfg["experiment"]
    seeds = experiment["training_seeds"]
    if seed is None:
        seed = seeds[0]
    elif seed not in seeds:
        logger.warning("seed %d is not in experiment.training_seeds %s - running anyway", seed, seeds)

    exp_targets = experiment.get("targets") or (experiment.get("matrix") or {}).get("targets")
    train_cfg = experiment["train"]
    fewshot = experiment.get("fewshot")
    composition = train_cfg.get("composition")
    matrix = experiment.get("matrix")
    test_mode = _test_language_mode(cfg)

    run_point: dict = {"seed": seed, "arm": None, "fewshot_size": None}

    if fewshot:
        arm = arm or "fewshot"
        if arm not in ("fewshot", "target_only"):
            raise ConfigError(f"unknown arm {arm!r} (expected 'fewshot' or 'target_only')")
        if arm == "target_only" and not fewshot.get("baseline_target_only", False):
            raise ConfigError("arm='target_only' requires experiment.fewshot.baseline_target_only=true")
        fewshot_size = fewshot_size or fewshot["sizes"][0]
        if fewshot_size not in fewshot["sizes"]:
            raise ConfigError(f"fewshot_size {fewshot_size} not in experiment.fewshot.sizes {fewshot['sizes']}")
        target = target or exp_targets[0]
        if target not in exp_targets:
            raise ConfigError(f"target {target!r} not in experiment.targets {exp_targets}")
        run_point.update({"arm": arm, "fewshot_size": fewshot_size, "train_language": "target"})
        run_point["train_langs"] = [target]
        run_point["eval_langs"] = [target] if test_mode == "train" else exp_targets
        run_point["val_langs"] = [target]
    elif composition:
        run_point["train_language"] = "source"
        run_point["train_langs"] = sorted(composition)
        run_point["eval_langs"] = exp_targets
        run_point["val_langs"] = sorted(composition)
    elif train_cfg.get("language") == "source":
        sources = (matrix or {}).get("sources", SOURCE_LANGUAGES)
        source = source or sources[0]
        if source not in sources:
            raise ConfigError(f"source {source!r} not in experiment.matrix.sources {sources}")
        run_point.update({"train_language": "source", "source": source})
        run_point["train_langs"] = [source]
        run_point["eval_langs"] = exp_targets
        run_point["val_langs"] = [source]
    else:  # training on a target language (EXP-0 style)
        target = target or exp_targets[0]
        if target not in exp_targets:
            raise ConfigError(f"target {target!r} not in experiment.targets {exp_targets}")
        run_point.update({"train_language": "target", "train_target": target})
        run_point["train_langs"] = [target]
        run_point["eval_langs"] = [target] if test_mode == "train" else exp_targets
        run_point["val_langs"] = [target]

    run_point["target_validation_langs"] = [
        t for t in exp_targets
        if cfg["evaluation"].get("eval_target_validation", True) and t not in run_point["val_langs"]
    ]
    return run_point


def variant_tags(cfg: dict, run_point: dict) -> dict:
    """Identity tags used to group runs during seed aggregation."""
    experiment = cfg["experiment"]
    tags: dict = {}
    if experiment.get("fewshot"):
        tags["arm"] = run_point["arm"]
        tags["fewshot_size"] = run_point["fewshot_size"]
        if run_point["arm"] == "fewshot":
            tags["init_experiment"] = experiment["init_from"]["experiment"]
            tags["init_source"] = experiment["init_from"]["source"]
    elif experiment["train"].get("composition"):
        tags["composition"] = experiment["train"]["composition"]
    elif run_point.get("train_language") == "source":
        tags["source"] = run_point.get("source")
        tags["train_size"] = experiment["train"]["size"]
    else:
        tags["train_language"] = run_point.get("train_target")
        tags["train_size"] = experiment["train"].get("size")
    if cfg["data"]["dataset"]["name"] == "synthetic":
        tags["dataset"] = "synthetic"
    return tags


# --------------------------------------------------------------------------
# data preparation
# --------------------------------------------------------------------------

def _repo_path(value) -> Path:
    path = Path(value)
    if not path.is_absolute():
        from axner.utils.paths import resolve_repo_root

        path = resolve_repo_root() / path
    return path


def _add_sentence_ids(ds, prefix: str):
    ids = [f"{prefix}-{i:06d}" for i in range(ds.num_rows)]
    return ds.add_column("sentence_id", ids)


def _prepare_synthetic(cfg: dict, run_point: dict) -> dict:
    from axner.data.stats import dataset_statistics, duplicates_report
    from axner.data.synthetic import build_synthetic_rows, synthetic_to_hf
    from axner.data.validation import check_label_schema, scan_iob2_issues

    synthetic = cfg["data"]["synthetic"]
    base_seed = synthetic["seed"]
    prepared: dict = {"test": {}, "target_validation": {}, "records": {"sampling": []}}

    def build(split_key, n, label):
        seed = derive_seed(base_seed, "generate", label)
        rows = build_synthetic_rows(n, seed)
        ds = synthetic_to_hf(rows).add_column("lang", ["synthetic"] * n)
        ds = _add_sentence_ids(ds, f"synthetic-{label}")
        prepared["records"]["sampling"].append({
            "language": "synthetic", "split": label, "n_generated": n, "seed": seed,
            "method": "synthetic_seeded_generation",
        })
        return ds

    prepared["train"] = build("train", synthetic["n_train"], "train")
    prepared["validation"] = build("validation", synthetic["n_validation"], "validation")
    for lang in run_point["eval_langs"]:
        prepared["test"][lang] = build(f"test_{lang}", synthetic["n_test"], f"test_{lang}")
    for lang in run_point["target_validation_langs"]:
        prepared["target_validation"][lang] = build(f"val_{lang}", synthetic["n_validation"], f"val_{lang}")

    # Audit, duplicate, and statistics records with the same shape as real data.
    label_names = list(LABEL_NAMES)
    report_only = {"policy": "report_only", "dropped": 0}
    splits_for_duplicates = {
        "train": prepared["train"],
        "validation": prepared["validation"],
        **{f"test_{lang}": ds for lang, ds in prepared["test"].items()},
    }
    prepared["records"]["audit"] = {
        "train": {"scan": scan_iob2_issues(prepared["train"], label_names), "policy_applied": report_only},
        "validation": {"scan": scan_iob2_issues(prepared["validation"], label_names), "policy_applied": report_only},
        "test": {lang: {"scan": scan_iob2_issues(ds, label_names), "policy_applied": report_only}
                 for lang, ds in prepared["test"].items()},
        "target_validation": {lang: {"scan": scan_iob2_issues(ds, label_names), "policy_applied": report_only}
                              for lang, ds in prepared["target_validation"].items()},
        "label_schema": check_label_schema(prepared["train"], label_names),
    }
    prepared["records"]["duplicates"] = duplicates_report(splits_for_duplicates)
    prepared["records"]["stats"] = {
        "train": dataset_statistics(prepared["train"], label_names, "train"),
        "validation": dataset_statistics(prepared["validation"], label_names, "validation"),
        **{f"test_{lang}": dataset_statistics(ds, label_names, f"test_{lang}")
           for lang, ds in prepared["test"].items()},
    }
    return prepared


def _load_split(cfg: dict, lang: str, split: str):
    from axner.data.wikiann import load_wikiann

    data_cfg = cfg["data"]
    return load_wikiann(
        lang,
        data_cfg["splits"][split],
        cache_dir=str(_repo_path(data_cfg["dataset"].get("cache_dir") or "data/hf_cache")),
        revision=data_cfg["dataset"].get("revision"),
    )


def _check_schema(ds, lang: str, split: str):
    schema = check_label_schema(ds, LABEL_NAMES)
    if not schema["same_set"]:
        raise DataValidationError(
            f"{lang}/{split}: dataset label names {schema['dataset_label_names']} do not match "
            f"the canonical label set {schema['expected_label_names']}"
        )
    return schema


def _resolve_train_size(cfg: dict, run_point: dict) -> int:
    train_cfg = cfg["experiment"]["train"]
    if run_point.get("fewshot_size"):
        return int(run_point["fewshot_size"])
    size = train_cfg.get("size")
    if size is None:
        raise ConfigError("experiment.train.size is required for non-fewshot runs")
    return int(size)


def _prepare_real(cfg: dict, run_point: dict) -> dict:
    data_cfg = cfg["data"]
    label_names = list(LABEL_NAMES)
    issue_policy = data_cfg["invalid_annotations"]["policy"]
    dup_policy = data_cfg["duplicates"]["policy"]
    prepared: dict = {"test": {}, "target_validation": {}, "records": {}}
    loaded: dict = {}

    def load(lang: str, split: str):
        key = (lang, split)
        if key not in loaded:
            ds = _load_split(cfg, lang, split)
            _check_schema(ds, lang, split)
            loaded[key] = ds
        return loaded[key]

    def scan_and_policy(ds, lang: str, split: str, enforce_policy: bool):
        scan = scan_iob2_issues(ds, label_names)
        if enforce_policy and scan["n_invalid_examples"]:
            ds, applied = apply_invalid_policy(ds, scan, issue_policy, split_name=f"{lang}/{split}")
        else:
            applied = {"policy": "report_only (test split untouched)" if not enforce_policy else issue_policy,
                       "dropped": 0}
        return ds, {"scan": scan, "policy_applied": applied}

    # ----- training data -----
    parts = []
    train_ds = None
    train_size = _resolve_train_size(cfg, run_point)
    fewshot = cfg["experiment"].get("fewshot")
    composition = cfg["experiment"]["train"].get("composition")
    if fewshot:
        # Nested few-shot subsets: sample the largest size once per seed, then
        # take prefixes so that size 10 ⊂ size 50 ⊂ size 100 for a given seed.
        n_max = max(fewshot["sizes"])
        lang = run_point["train_langs"][0]
        seed = derive_seed(cfg["experiment"]["sampling_seed"], "fewshot", lang, str(n_max))
        indices, record, sampled = sample_split(load(lang, "train"), n_max, seed, lang, "train")
        indices = indices[:train_size]
        sampled = sampled.select(indices)
        record.update({"n_requested": train_size, "n_sampled": len(indices), "nested_from_size": n_max,
                       "indices_sha256": None, "indices": indices})
        parts.append((lang, sampled, len(indices)))
        prepared["records"]["sampling"] = [record]
    elif composition:
        raw_parts = [
            (lang, load(lang, "train"), composition[lang])
            for lang in sorted(composition)
        ]
        combined, comp_record = compose_multisource(
            raw_parts, cfg["experiment"]["sampling_seed"], shuffle=True
        )
        prepared["records"]["sampling"] = comp_record["sources"]
        prepared["records"]["composition"] = {
            k: v for k, v in comp_record.items() if k != "sources"
        }
        train_ds = combined
    else:
        lang = run_point["train_langs"][0]
        seed = language_sampling_seed(cfg["experiment"]["sampling_seed"], lang, "train")
        indices, record, sampled = sample_split(load(lang, "train"), train_size, seed, lang, "train")
        prepared["records"]["sampling"] = [record]
        parts.append((lang, sampled, train_size))

    if parts:
        import datasets as hf_datasets

        encoded = [ds.add_column("lang", [lang] * ds.num_rows) for lang, ds, _ in parts]
        train_ds = hf_datasets.concatenate_datasets(encoded)

    train_ds, train_audit = scan_and_policy(train_ds, "+".join(run_point["train_langs"]), "train", True)

    # ----- validation data (used during training / checkpoint selection) -----
    val_cfg = cfg["experiment"].get("validation", {})
    val_language = val_cfg.get("language") or cfg["evaluation"]["validation_language"]
    val_parts = []
    for lang in run_point["val_langs"]:
        ds = load(lang, "validation")
        size = val_cfg.get("size") if val_language == "target" else cfg["evaluation"].get("validation_size_per_language")
        if val_language == "target" and size is None:
            sampled = ds
        else:
            size = size if size is not None else ds.num_rows
            seed = language_sampling_seed(cfg["experiment"]["sampling_seed"], lang, "validation")
            _, record, sampled = sample_split(ds, min(size, ds.num_rows), seed, lang, "validation")
            prepared["records"]["sampling"].append(record)
        val_parts.append(sampled.add_column("lang", [lang] * sampled.num_rows))
    validation_ds = val_parts[0] if len(val_parts) == 1 else _concat(val_parts)
    validation_ds, val_audit = scan_and_policy(validation_ds, "validation", "validation", True)

    # ----- test data (full target test splits; never altered) -----
    test_audits = {}
    test_ds = {}
    for lang in run_point["eval_langs"]:
        ds = load(lang, "test")
        ds, audit = scan_and_policy(ds, lang, "test", enforce_policy=False)
        test_ds[lang] = ds
        test_audits[lang] = audit

    # ----- target validation splits (EXP-3 selection support; never test data) -----
    target_val_audits = {}
    for lang in run_point["target_validation_langs"]:
        ds = load(lang, "validation")
        ds, audit = scan_and_policy(ds, lang, "validation", enforce_policy=False)
        target_val_ds = ds.add_column("lang", [lang] * ds.num_rows)
        prepared["target_validation"][lang] = _add_sentence_ids(target_val_ds, f"{lang}-validation")
        target_val_audits[lang] = audit

    # ----- ids, duplicates, statistics -----
    train_ds = _add_sentence_ids(train_ds, f"train-{'+'.join(run_point['train_langs'])}")
    validation_ds = _add_sentence_ids(validation_ds, "validation")
    for lang, ds in test_ds.items():
        test_ds[lang] = _add_sentence_ids(ds, f"{lang}-test")

    splits_for_duplicates = {"train": train_ds, "validation": validation_ds}
    for lang, ds in test_ds.items():
        splits_for_duplicates[f"test_{lang}"] = ds
    duplicates = duplicates_report(splits_for_duplicates)
    dropped_from_duplicates = 0
    if dup_policy == "drop_train_dups":
        test_keys = set()
        for lang, ds in test_ds.items():
            test_keys.update(tuple(row["tokens"]) for row in ds)
        test_keys.update(tuple(row["tokens"]) for row in validation_ds)
        keep = [
            i for i, row in enumerate(train_ds)
            if tuple(row["tokens"]) not in test_keys
        ]
        dropped_from_duplicates = train_ds.num_rows - len(keep)
        if dropped_from_duplicates:
            logger.warning("dropped %d training rows duplicating val/test token sequences", dropped_from_duplicates)
            train_ds = train_ds.select(keep)
        duplicates["policy_applied"] = {"policy": dup_policy, "dropped_from_train": dropped_from_duplicates}
    else:
        duplicates["policy_applied"] = {"policy": dup_policy, "dropped_from_train": 0}

    stats = {
        "train": dataset_statistics(train_ds, label_names, "train"),
        "validation": dataset_statistics(validation_ds, label_names, "validation"),
    }
    for lang, ds in test_ds.items():
        stats[f"test_{lang}"] = dataset_statistics(ds, label_names, f"test_{lang}")

    prepared["train"] = train_ds
    prepared["validation"] = validation_ds
    prepared["test"] = test_ds
    prepared["records"]["audit"] = {
        "train": train_audit,
        "validation": val_audit,
        "test": test_audits,
        "target_validation": target_val_audits,
        "label_schema": check_label_schema(next(iter(loaded.values())), LABEL_NAMES),
    }
    prepared["records"]["duplicates"] = duplicates
    prepared["records"]["stats"] = stats
    return prepared


def _concat(parts):
    import datasets as hf_datasets

    return hf_datasets.concatenate_datasets(parts)


def prepare_data(cfg: dict, run_point: dict) -> dict:
    """Build train/validation/test datasets for the run point."""
    if cfg["data"]["dataset"]["name"] == "synthetic":
        return _prepare_synthetic(cfg, run_point)
    return _prepare_real(cfg, run_point)


# --------------------------------------------------------------------------
# EXP-3 checkpoint resolution (target-validation based, never test)
# --------------------------------------------------------------------------

def resolve_init_checkpoint(cfg: dict, run_point: dict, outputs_root) -> Path:
    """Resolve the EXP-3 init checkpoint from explicit config or artifacts."""
    init_from = cfg["experiment"]["init_from"]
    explicit = init_from.get("checkpoint")
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = _repo_path(path)
        if not path.exists():
            raise ArtifactError(f"init_from.checkpoint does not exist: {path}")
        return path

    from axner.evaluation.aggregation import best_run_by

    experiment_id = init_from["experiment"]
    source = init_from["source"]
    metric = init_from["selection_metric"]
    if metric != "target_validation_entity_f1":
        raise ConfigError(f"unsupported selection metric {metric!r}")

    exp_dir = Path(outputs_root) / experiment_id
    if not exp_dir.exists():
        raise ArtifactError(
            f"no artifacts for experiment {experiment_id!r} under {outputs_root}; "
            "run EXP-1 first (or import Colab/Kaggle bundles WITH checkpoints)"
        )
    candidate_runs = []
    for run_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
        try:
            manifest = load_json(run_dir / "manifest.json")
        except FileNotFoundError:
            continue
        if manifest.get("variant_tags", {}).get("source") == source:
            candidate_runs.append(run_dir)
    if not candidate_runs:
        raise ArtifactError(
            f"no {experiment_id} runs with source={source!r} under {outputs_root}"
        )
    target = run_point["train_langs"][0]
    best = best_run_by(candidate_runs, f"validation_target_{target}.json", "entity_f1")
    if best is None:
        raise ArtifactError(
            f"no validation_target_{target}.json metrics found in {experiment_id} runs; "
            "ensure evaluation.eval_target_validation was enabled"
        )
    run_dir, value = best
    checkpoint_dir = Path(run_dir) / "checkpoints"
    if not any(checkpoint_dir.iterdir()):
        raise ArtifactError(
            f"best run {run_dir} has no checkpoints directory content; "
            "import the Colab/Kaggle bundle with checkpoints (scripts/import_artifacts.py)"
        )
    logger.info("selected init checkpoint from %s (target val entity_f1=%.4f)", run_dir, value)
    return checkpoint_dir


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------

def execute_run(config, seed=None, target=None, source=None, arm=None, fewshot_size=None,
                output_base=None, run_label=None, dry_run=False) -> dict:
    """Execute one run end-to-end and return a summary dict.

    ``config`` is an experiment id/path (loaded via :func:`load_run_config`)
    or an already-validated merged config dict.
    """
    if isinstance(config, (str, Path)):
        cfg = load_run_config(config)
    else:
        cfg = validate_or_raise(config)

    run_point = resolve_run_point(cfg, seed=seed, target=target, source=source,
                                  arm=arm, fewshot_size=fewshot_size)
    seed = run_point["seed"]
    experiment_id = cfg["experiment"]["id"]
    if output_base is None:
        output_base = _repo_path(cfg["project"]["paths"]["outputs_dir"])
    run_dir = create_run_directory(output_base, experiment_id, label=run_label)
    paths = artifact_paths(run_dir)
    tags = variant_tags(cfg, run_point)

    logger.info("run directory: %s", run_dir)
    save_yaml(cfg, paths["config_resolved"])
    env_info = capture_environment_info()
    save_json(paths["environment"], env_info)
    write_manifest(
        run_dir,
        experiment_id=experiment_id,
        training_seed=seed,
        variant_tags=tags,
        model_id=("tiny_random_offline" if cfg["model"].get("tiny_random")
                  else cfg["model"]["name"]),
        dataset=cfg["data"]["dataset"]["name"],
        dataset_revision=cfg["data"]["dataset"].get("revision"),
        sampling_seed=cfg["experiment"]["sampling_seed"],
        label_names=list(LABEL_NAMES),
        run_point={k: v for k, v in run_point.items() if k != "seed"},
        git_commit=(env_info.get("git") or {}).get("commit"),
        status="running",
    )

    prepared = prepare_data(cfg, run_point)
    save_json(paths["sampling"], prepared["records"].get("sampling", []))
    if prepared["records"].get("composition"):
        save_json(run_dir / "data" / "composition.json", prepared["records"]["composition"])
    save_json(paths["data_audit"], prepared["records"]["audit"])
    save_json(paths["duplicates"], prepared["records"]["duplicates"])
    save_json(paths["dataset_stats"], prepared["records"]["stats"])

    if dry_run:
        update_manifest(run_dir, status="dry_run_completed",
                        train_rows=prepared["train"].num_rows)
        logger.info("dry run complete; no model/training executed")
        return {"run_dir": str(run_dir), "status": "dry_run_completed",
                "train_rows": prepared["train"].num_rows,
                "test_rows": {lang: ds.num_rows for lang, ds in prepared["test"].items()}}

    label_names = list(LABEL_NAMES)
    compute_metrics = _build_compute_metrics(label_names)

    # ----- tokenizer -----
    model_cfg = cfg["model"]
    if model_cfg.get("tiny_random"):
        from axner.models.offline import WhitespaceTokenizer

        tokenizer = WhitespaceTokenizer(vocab_size=512, max_length=model_cfg["max_length"])
        pad_token_id = WhitespaceTokenizer.pad_token_id
        logger.warning("using offline stub tokenizer (pipeline validation only)")
    else:
        from axner.models.init import init_tokenizer

        tokenizer = init_tokenizer(model_cfg["name"], cache_dir=str(_repo_path(model_cfg.get("cache_dir") or "data/hf_cache")))
        pad_token_id = tokenizer.pad_token_id or 0

    # ----- alignment -----
    aligned_train = align_dataset(prepared["train"], tokenizer, model_cfg["max_length"])
    aligned_val = align_dataset(prepared["validation"], tokenizer, model_cfg["max_length"])
    aligned_test = {lang: align_dataset(ds, tokenizer, model_cfg["max_length"])
                    for lang, ds in prepared["test"].items()}
    aligned_target_val = {lang: align_dataset(ds, tokenizer, model_cfg["max_length"])
                          for lang, ds in prepared["target_validation"].items()}
    truncation = {
        "train": truncation_stats(aligned_train),
        "validation": truncation_stats(aligned_val),
        "test": {lang: truncation_stats(ds) for lang, ds in aligned_test.items()},
    }
    save_json(run_dir / "data" / "truncation_stats.json", truncation)

    # ----- model -----
    from axner.models.init import init_model

    fewshot_arm = run_point.get("arm")
    init_ckpt = None
    if fewshot_arm == "fewshot":
        init_ckpt = resolve_init_checkpoint(cfg, run_point, output_base)
        model = init_model(str(init_ckpt), label_names, seed=seed,
                           cache_dir=str(_repo_path(model_cfg.get("cache_dir") or "data/hf_cache")))
    else:
        model = init_model(model_cfg["name"], label_names, seed=seed,
                           cache_dir=str(_repo_path(model_cfg.get("cache_dir") or "data/hf_cache")),
                           tiny_random=model_cfg.get("tiny_random", False))

    training_cfg = cfg["training"]
    training_cfg = {**training_cfg}
    if fewshot_arm:
        training_cfg["_eval_output_dir"] = str(run_dir / "checkpoints")
        training_cfg["_seed"] = seed

    best_checkpoint = None
    if training_cfg.get("enabled", True):
        from axner.models.training import best_checkpoint_path, run_training, trainer_log_history

        trainer = run_training(
            model, tokenizer, aligned_train, aligned_val, run_dir / "checkpoints",
            seed, training_cfg, compute_metrics, pad_token_id=pad_token_id,
        )
        save_json(paths["trainer_history"], trainer_log_history(trainer))
        val_metrics = {k: v for k, v in trainer.evaluate(aligned_val).items()
                       if not k.startswith("eval_runtime") and not k.startswith("eval_samples")}
        save_json(paths["validation_metrics"], val_metrics)
        best_checkpoint = best_checkpoint_path(trainer)
        model = trainer.model
        predict_trainer = trainer
    else:
        from axner.models.training import make_eval_trainer

        predict_trainer = make_eval_trainer(
            model, tokenizer, aligned_val, training_cfg, compute_metrics, pad_token_id=pad_token_id
        )
        logger.info("training disabled (training.enabled=false); evaluation only")

    if best_checkpoint:
        update_manifest(run_dir, best_checkpoint=str(best_checkpoint))

    # ----- evaluation on each target test split -----
    evaluation_cfg = cfg["evaluation"]
    bootstrap_cfg = evaluation_cfg["bootstrap"]
    summary_metrics = {}
    for lang, ds in aligned_test.items():
        from axner.models.training import predict_dataset

        predictions, label_ids, _ = predict_dataset(predict_trainer, ds)
        sentence_records = build_sentence_records(ds, predictions, label_ids, label_names)
        gold_seqs = [r["gold_tags"] for r in sentence_records]
        pred_seqs = [r["pred_tags"] for r in sentence_records]
        metrics = compute_entity_metrics(gold_seqs, pred_seqs)
        metrics["per_type"] = per_type_metrics(gold_seqs, pred_seqs)
        metrics["error_summary"] = error_summary(gold_seqs, pred_seqs)
        save_json(run_dir / "metrics" / f"test_metrics_{lang}.json", metrics)
        if evaluation_cfg["export"].get("sentence_level", True):
            export_jsonl(sentence_records, str(paths["sentence_level"]).format(lang=lang))
        if evaluation_cfg["export"].get("entity_level", True):
            export_jsonl(build_entity_records(sentence_records),
                         str(paths["entity_level"]).format(lang=lang))
        bootstrap = bootstrap_ci(
            gold_seqs, pred_seqs,
            n_resamples=bootstrap_cfg["n_resamples"],
            seed=derive_seed(bootstrap_cfg["seed"], lang),
            confidence_level=bootstrap_cfg["confidence_level"],
        )
        save_json(run_dir / "metrics" / f"bootstrap_{lang}.json", bootstrap)
        summary_metrics[lang] = {"entity_f1": metrics["f1"], "precision": metrics["precision"],
                                 "recall": metrics["recall"]}
        logger.info("target %s: entity_f1=%.4f (n_sentences=%d)", lang, metrics["f1"], metrics["n_sentences"])

    # ----- target validation evaluation (EXP-3 selection support) -----
    for lang, ds in aligned_target_val.items():
        from axner.models.training import predict_dataset

        predictions_t, label_ids_t, _ = predict_dataset(predict_trainer, ds)
        sentence_records = build_sentence_records(ds, predictions_t, label_ids_t, label_names)
        metrics = compute_entity_metrics(
            [r["gold_tags"] for r in sentence_records],
            [r["pred_tags"] for r in sentence_records],
        )
        save_json(run_dir / "metrics" / f"validation_target_{lang}.json", metrics)

    update_manifest(
        run_dir,
        status="completed",
        init_checkpoint=str(init_ckpt) if init_ckpt else None,
        train_rows=aligned_train.num_rows,
        eval_languages=list(summary_metrics),
        entity_f1_by_target={lang: m["entity_f1"] for lang, m in summary_metrics.items()},
    )
    return {
        "run_dir": str(run_dir),
        "status": "completed",
        "seed": seed,
        "variant_tags": tags,
        "entity_f1_by_target": {lang: m["entity_f1"] for lang, m in summary_metrics.items()},
    }


def _build_compute_metrics(label_names):
    from axner.models.training import build_compute_metrics

    return build_compute_metrics(label_names)
