"""Configuration validation.

``validate_config`` returns a list of human-readable problems; the empty list
means the configuration is valid. ``validate_or_raise`` turns the list into a
single :class:`ConfigError` containing every problem, so invalid configs fail
early with complete diagnostics instead of crashing deep inside a run.

Validation is intentionally strict about research-critical invariants
(label scheme, composition totals, seed lists) and tolerant about purely
cosmetic settings.
"""

from __future__ import annotations

from pathlib import Path

from axner.constants import (
    DATASET_NAMES,
    DUPLICATE_POLICIES,
    ENTITY_TYPES,
    ISSUE_POLICIES,
    LABEL_NAMES,
    LABEL_SCHEMES,
    SAMPLING_METHODS,
    SELECTION_METRICS,
    SUBWORD_POLICIES,
    SUPPORTED_LANGUAGES,
    TEST_LANGUAGE_MODES,
    VALIDATION_LANGUAGES,
)
from axner.errors import ConfigError


def _unique_languages(values) -> bool:
    return len(values) == len(set(values))


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_config(cfg: dict) -> list[str]:
    """Validate a merged configuration dict; return all problems found."""
    errors: list[str] = []

    def require(condition: bool, message: str) -> bool:
        if not condition:
            errors.append(message)
        return condition

    for section in ("project", "data", "model", "training", "evaluation", "experiment"):
        require(section in cfg, f"missing required configuration section '{section}'")
    if errors:
        return errors  # cannot validate contents of missing sections

    project = cfg["project"]
    data = cfg["data"]
    model = cfg["model"]
    training = cfg["training"]
    evaluation = cfg["evaluation"]
    experiment = cfg["experiment"]

    # ---------------- project ----------------
    require(
        project.get("label_scheme") in LABEL_SCHEMES,
        f"project.label_scheme must be one of {LABEL_SCHEMES}, got {project.get('label_scheme')!r}",
    )
    entity_types = project.get("entity_types")
    require(
        isinstance(entity_types, list)
        and len(entity_types) > 0
        and all(t in ENTITY_TYPES for t in entity_types),
        f"project.entity_types must be a non-empty subset of {ENTITY_TYPES}, got {entity_types!r}",
    )
    languages = project.get("languages", {})
    for role in ("sources", "targets"):
        values = languages.get(role)
        require(
            isinstance(values, list)
            and len(values) > 0
            and all(v in SUPPORTED_LANGUAGES for v in values)
            and _unique_languages(values),
            f"project.languages.{role} must be a non-empty list of unique supported languages "
            f"{SUPPORTED_LANGUAGES}, got {values!r}",
        )
    known = project.get("known_max_train_sizes", {})
    require(
        isinstance(known, dict) and all(_is_int(v) and v > 0 for v in known.values()),
        "project.known_max_train_sizes must map language codes to positive integers",
    )
    require(
        isinstance(project.get("known_max_sizes_verified", False), bool),
        "project.known_max_sizes_verified must be a boolean",
    )
    for key in ("outputs_dir", "data_dir"):
        value = project.get("paths", {}).get(key)
        require(
            isinstance(value, str) and value and not value.startswith("/"),
            f"project.paths.{key} must be a non-empty relative path, got {value!r}",
        )

    # ---------------- data ----------------
    dataset = data.get("dataset", {})
    require(
        dataset.get("name") in DATASET_NAMES,
        f"data.dataset.name must be one of {DATASET_NAMES}, got {dataset.get('name')!r}",
    )
    require(
        dataset.get("revision") is None or isinstance(dataset.get("revision"), str),
        "data.dataset.revision must be null or a string",
    )
    label_names = data.get("label_names")
    require(
        label_names == list(LABEL_NAMES),
        f"data.label_names must equal the canonical label list {list(LABEL_NAMES)}, got {label_names!r}",
    )
    splits = data.get("splits", {})
    for key in ("train", "validation", "test"):
        require(
            isinstance(splits.get(key), str) and splits.get(key),
            f"data.splits.{key} must be a non-empty string, got {splits.get(key)!r}",
        )
    sampling = data.get("sampling", {})
    require(
        sampling.get("method") in SAMPLING_METHODS,
        f"data.sampling.method must be one of {SAMPLING_METHODS}, got {sampling.get('method')!r}",
    )
    require(
        _is_int(sampling.get("seed")) and sampling["seed"] >= 0,
        f"data.sampling.seed must be a non-negative integer, got {sampling.get('seed')!r}",
    )
    require(
        data.get("invalid_annotations", {}).get("policy") in ISSUE_POLICIES,
        f"data.invalid_annotations.policy must be one of {ISSUE_POLICIES}, got "
        f"{data.get('invalid_annotations', {}).get('policy')!r}",
    )
    require(
        data.get("duplicates", {}).get("policy") in DUPLICATE_POLICIES,
        f"data.duplicates.policy must be one of {DUPLICATE_POLICIES}, got "
        f"{data.get('duplicates', {}).get('policy')!r}",
    )
    synthetic = data.get("synthetic")
    if dataset.get("name") == "synthetic":
        require(
            isinstance(synthetic, dict)
            and all(_is_int(synthetic.get(k)) and synthetic[k] >= 0
                    for k in ("n_train", "n_validation", "n_test")),
            "data.synthetic must define non-negative integers n_train, n_validation, n_test",
        )
        require(
            synthetic is not None and _is_int(synthetic.get("seed")),
            "data.synthetic.seed must be an integer",
        )

    # ---------------- model ----------------
    require(
        isinstance(model.get("name"), str) and model.get("name"),
        "model.name must be a non-empty string (Hugging Face model id)",
    )
    max_length = model.get("max_length")
    require(
        _is_int(max_length) and 16 <= max_length <= 2048,
        f"model.max_length must be an integer in [16, 2048], got {max_length!r}",
    )
    require(
        model.get("subword_policy") in SUBWORD_POLICIES,
        f"model.subword_policy must be one of {SUBWORD_POLICIES}, got {model.get('subword_policy')!r}",
    )
    require(
        isinstance(model.get("tiny_random", False), bool),
        "model.tiny_random must be a boolean",
    )

    # ---------------- training ----------------
    require(
        isinstance(training.get("enabled", True), bool),
        "training.enabled must be a boolean",
    )
    hp = training.get("hyperparameters", {})
    lr = hp.get("learning_rate")
    require(
        isinstance(lr, (int, float)) and not isinstance(lr, bool) and 0 < lr <= 1,
        f"training.hyperparameters.learning_rate must be a float in (0, 1], got {lr!r}",
    )
    batch = hp.get("batch_size")
    require(
        _is_int(batch) and 1 <= batch <= 256,
        f"training.hyperparameters.batch_size must be an integer in [1, 256], got {batch!r}",
    )
    epochs = hp.get("epochs")
    require(
        _is_int(epochs) and 1 <= epochs <= 100,
        f"training.hyperparameters.epochs must be an integer in [1, 100], got {epochs!r}",
    )
    wd = hp.get("weight_decay")
    require(
        isinstance(wd, (int, float)) and not isinstance(wd, bool) and wd >= 0,
        f"training.hyperparameters.weight_decay must be >= 0, got {wd!r}",
    )
    warmup = hp.get("warmup_ratio")
    require(
        isinstance(warmup, (int, float)) and not isinstance(warmup, bool) and 0 <= warmup <= 1,
        f"training.hyperparameters.warmup_ratio must be in [0, 1], got {warmup!r}",
    )
    gn = hp.get("max_grad_norm")
    require(
        isinstance(gn, (int, float)) and not isinstance(gn, bool) and gn > 0,
        f"training.hyperparameters.max_grad_norm must be > 0, got {gn!r}",
    )
    eval_batch = training.get("eval_batch_size")
    require(
        _is_int(eval_batch) and 1 <= eval_batch <= 256,
        f"training.eval_batch_size must be an integer in [1, 256], got {eval_batch!r}",
    )
    for key in ("fp16", "deterministic"):
        require(
            isinstance(training.get(key, False), bool),
            f"training.{key} must be a boolean",
        )
    require(
        training.get("device") in ("auto", "cpu", "cuda"),
        f"training.device must be one of ['auto', 'cpu', 'cuda'], got {training.get('device')!r}",
    )
    checkpoint = training.get("checkpoint", {})
    require(
        isinstance(checkpoint.get("selection_metric"), str) and checkpoint.get("selection_metric"),
        "training.checkpoint.selection_metric must be a non-empty string",
    )
    limit = checkpoint.get("save_total_limit")
    require(
        _is_int(limit) and limit >= 1,
        f"training.checkpoint.save_total_limit must be an integer >= 1, got {limit!r}",
    )

    # ---------------- evaluation ----------------
    require(
        evaluation.get("validation_language") in VALIDATION_LANGUAGES,
        f"evaluation.validation_language must be one of {VALIDATION_LANGUAGES}, got "
        f"{evaluation.get('validation_language')!r}",
    )
    vsize = evaluation.get("validation_size_per_language")
    require(
        vsize is None or (_is_int(vsize) and vsize >= 1),
        f"evaluation.validation_size_per_language must be null or an integer >= 1, got {vsize!r}",
    )
    require(
        evaluation.get("test_language_mode") in TEST_LANGUAGE_MODES,
        f"evaluation.test_language_mode must be one of {TEST_LANGUAGE_MODES}, got "
        f"{evaluation.get('test_language_mode')!r}",
    )
    require(
        isinstance(evaluation.get("eval_target_validation", True), bool),
        "evaluation.eval_target_validation must be a boolean",
    )
    bootstrap = evaluation.get("bootstrap", {})
    n_resamples = bootstrap.get("n_resamples")
    require(
        _is_int(n_resamples) and 100 <= n_resamples <= 1_000_000,
        f"evaluation.bootstrap.n_resamples must be an integer in [100, 1000000], got {n_resamples!r}",
    )
    require(
        _is_int(bootstrap.get("seed")),
        f"evaluation.bootstrap.seed must be an integer, got {bootstrap.get('seed')!r}",
    )
    level = bootstrap.get("confidence_level")
    require(
        isinstance(level, (int, float)) and not isinstance(level, bool) and 0 < level < 1,
        f"evaluation.bootstrap.confidence_level must be a float in (0, 1), got {level!r}",
    )
    for key in ("sentence_level", "entity_level"):
        require(
            isinstance(evaluation.get("export", {}).get(key, True), bool),
            f"evaluation.export.{key} must be a boolean",
        )

    # ---------------- experiment ----------------
    require(
        isinstance(experiment.get("id"), str) and experiment.get("id"),
        "experiment.id must be a non-empty string",
    )
    seeds = experiment.get("training_seeds")
    require(
        isinstance(seeds, list) and len(seeds) > 0
        and all(_is_int(s) and s >= 0 for s in seeds)
        and _unique_languages(seeds),
        f"experiment.training_seeds must be a non-empty list of unique non-negative integers, got {seeds!r}",
    )
    require(
        _is_int(experiment.get("sampling_seed")),
        f"experiment.sampling_seed must be an integer, got {experiment.get('sampling_seed')!r}",
    )
    exp_targets = experiment.get("targets")
    matrix = experiment.get("matrix")
    if exp_targets is None and matrix:
        exp_targets = (matrix or {}).get("targets")  # matrix experiments define targets there
    require(
        isinstance(exp_targets, list) and len(exp_targets) > 0
        and all(t in SUPPORTED_LANGUAGES for t in exp_targets)
        and _unique_languages(exp_targets),
        f"experiment.targets (or experiment.matrix.targets) must be a non-empty list of unique "
        f"supported languages, got {experiment.get('targets')!r}",
    )

    train_cfg = experiment.get("train", {})
    train_language = train_cfg.get("language")
    require(
        train_language in ("source", "target", None),
        f"experiment.train.language must be 'source' or 'target', got {train_language!r}",
    )
    composition = train_cfg.get("composition")
    matrix = experiment.get("matrix")
    require(
        not (composition and matrix),
        "experiment.train.composition and experiment.matrix are mutually exclusive",
    )
    require(
        not (composition and matrix is None and train_language == "source" and train_cfg.get("size") is not None),
        "single-source experiments (train.language=source with a fixed size) require experiment.matrix.sources; "
        "multi-language training data requires experiment.train.composition",
    )
    if matrix:
        require(
            isinstance(matrix, dict)
            and isinstance(matrix.get("sources"), list) and matrix.get("sources")
            and all(s in SUPPORTED_LANGUAGES for s in matrix["sources"])
            and _unique_languages(matrix["sources"]),
            f"experiment.matrix.sources must be a non-empty list of unique supported languages, got {matrix!r}",
        )
        require(
            isinstance(matrix.get("targets"), list) and matrix.get("targets")
            and all(t in SUPPORTED_LANGUAGES for t in matrix["targets"])
            and _unique_languages(matrix["targets"]),
            f"experiment.matrix.targets must be a non-empty list of unique supported languages, got {matrix!r}",
        )
        if errors == [] or all("matrix" not in e for e in errors):
            m_sources = matrix.get("sources") or []
            m_targets = matrix.get("targets") or []
            overlap = set(m_sources) & set(m_targets)
            require(
                not overlap,
                f"experiment.matrix sources and targets must be disjoint for zero-shot transfer, "
                f"found overlap: {sorted(overlap)}",
            )
    if composition:
        require(
            isinstance(composition, dict) and composition
            and all(k in SUPPORTED_LANGUAGES for k in composition)
            and all(_is_int(v) and v > 0 for v in composition.values()),
            f"experiment.train.composition must map supported languages to positive integers, got {composition!r}",
        )
        total = train_cfg.get("size")
        if isinstance(composition, dict) and composition and _is_int(total):
            comp_sum = sum(composition.values())
            require(
                comp_sum == total,
                f"experiment.train.composition must sum exactly to experiment.train.size: "
                f"{comp_sum} != {total} (composition {composition})",
            )

    size = train_cfg.get("size")
    is_synthetic = dataset.get("name") == "synthetic"
    fewshot = experiment.get("fewshot")
    if size is not None:
        require(
            _is_int(size) and size > 0,
            f"experiment.train.size must be null or a positive integer, got {size!r}",
        )
        if (
            _is_int(size)
            and not is_synthetic
            and train_language == "target"
            and not fewshot
            and exp_targets
            and len(exp_targets) == 1
        ):
            available = project.get("known_max_train_sizes", {}).get(exp_targets[0])
            if available is not None and size > available:
                require(
                    False,
                    f"experiment.train.size {size} exceeds recorded WikiANN training size {available} "
                    f"for language {exp_targets[0]!r}",
                )
    elif not is_synthetic and not fewshot:
        require(
            False,
            "experiment.train.size must be set unless the dataset is synthetic or fewshot sizes are used",
        )

    exp_validation = experiment.get("validation", {})
    val_language = exp_validation.get("language") or evaluation.get("validation_language")
    require(
        val_language in VALIDATION_LANGUAGES,
        f"experiment.validation.language must be one of {VALIDATION_LANGUAGES}, got {val_language!r}",
    )
    val_size = exp_validation.get("size")
    require(
        val_size is None or (_is_int(val_size) and val_size >= 1),
        f"experiment.validation.size must be null or a positive integer, got {val_size!r}",
    )

    require(
        experiment.get("test_language_mode", evaluation.get("test_language_mode")) in TEST_LANGUAGE_MODES,
        f"experiment.test_language_mode must be one of {TEST_LANGUAGE_MODES}",
    )

    if fewshot:
        sizes = fewshot.get("sizes")
        require(
            isinstance(sizes, list) and sizes
            and all(_is_int(s) and s > 0 for s in sizes)
            and sizes == sorted(sizes),
            "experiment.fewshot.sizes must be a non-empty ascending list of positive integers, got {sizes!r}",
        )
        require(
            isinstance(fewshot.get("baseline_target_only", False), bool),
            "experiment.fewshot.baseline_target_only must be a boolean",
        )

    init_from = experiment.get("init_from")
    if init_from:
        require(
            isinstance(init_from, dict) and isinstance(init_from.get("experiment"), str)
            and init_from.get("experiment"),
            "experiment.init_from.experiment must be a non-empty string",
        )
        require(
            init_from.get("source") in SUPPORTED_LANGUAGES,
            f"experiment.init_from.source must be a supported language, got {init_from.get('source')!r}",
        )
        require(
            init_from.get("selection_metric") in SELECTION_METRICS,
            f"experiment.init_from.selection_metric must be one of {SELECTION_METRICS}, got "
            f"{init_from.get('selection_metric')!r}",
        )
        ckpt = init_from.get("checkpoint")
        if ckpt is not None:
            require(
                isinstance(ckpt, str) and Path(ckpt).exists(),
                f"experiment.init_from.checkpoint must be null or an existing directory path, got {ckpt!r}",
            )
        require(
            init_from.get("source") not in (exp_targets or []),
            "experiment.init_from.source must not be a target language of this experiment",
        )

    return errors


def validate_or_raise(cfg: dict) -> dict:
    """Validate a merged configuration; raise :class:`ConfigError` listing all problems."""
    errors = validate_config(cfg)
    if errors:
        header = f"configuration validation failed with {len(errors)} problem(s):"
        raise ConfigError(header + "\n" + "\n".join(f"  - {e}" for e in errors))
    return cfg
