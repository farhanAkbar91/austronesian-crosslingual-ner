"""End-to-end pipeline smoke test with the offline tiny model (torch required).

Runs the full runner on synthetic data with a tiny randomly initialized
model: alignment -> training -> validation-based checkpoint selection ->
test prediction -> metrics, bootstrap, and prediction export. Marks:
    pytest -m torch
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from axner.artifacts.manager import load_json, load_jsonl, load_manifest  # noqa: E402
from axner.experiments.runner import execute_run  # noqa: E402
from axner.models.init import init_model  # noqa: E402
from axner.models.offline import WhitespaceTokenizer  # noqa: E402
from axner.models.training import build_compute_metrics, predict_dataset, run_training  # noqa: E402


pytestmark = pytest.mark.torch


def test_training_module_end_to_end(synthetic_dataset, stub_tokenizer, label_names, tmp_path):
    import datasets as hf_datasets

    train_ds = synthetic_dataset.select(range(32))
    val_ds = synthetic_dataset.select(range(32, 40))
    aligned_train = hf_datasets.Dataset.from_list([
        {**row} for row in _align_rows(train_ds, stub_tokenizer)
    ])
    aligned_val = hf_datasets.Dataset.from_list(_align_rows(val_ds, stub_tokenizer))

    model = init_model("tiny_random_offline", label_names, seed=42, tiny_random=True)
    training_cfg = {
        "hyperparameters": {
            "learning_rate": 5e-3, "batch_size": 8, "epochs": 1,
            "weight_decay": 0.0, "warmup_ratio": 0.0, "max_grad_norm": 1.0,
        },
        "eval_batch_size": 8,
        "fp16": False,
        "deterministic": True,
        "logging_steps": 1,
        "device": "cpu",
        "checkpoint": {"selection_metric": "entity_f1", "save_total_limit": 1},
    }
    trainer = run_training(
        model, stub_tokenizer, aligned_train, aligned_val,
        output_dir=tmp_path / "checkpoints", seed=42,
        training_cfg=training_cfg,
        compute_metrics=build_compute_metrics(label_names),
        pad_token_id=WhitespaceTokenizer.pad_token_id,
    )
    assert trainer.state.best_model_checkpoint is not None

    predictions, label_ids, _ = predict_dataset(trainer, aligned_val)
    assert predictions.shape[0] == aligned_val.num_rows
    assert label_ids.shape == predictions.shape[:2]


def _align_rows(ds, tokenizer):
    from axner.data.alignment import align_tokens_labels

    aligned = []
    for row in ds:
        result = align_tokens_labels(tokenizer, row["tokens"], row["ner_tags"], max_length=32)
        aligned.append({**row, **result})
    return aligned


def test_execute_run_full_smoke(synthetic_config, tmp_path):
    summary = execute_run(synthetic_config, output_base=tmp_path, run_label="torchsmoke")
    assert summary["status"] == "completed"

    from pathlib import Path

    run_dir = Path(summary["run_dir"])
    manifest = load_manifest(run_dir)
    assert manifest["status"] == "completed"
    assert manifest["model_id"] == "tiny_random_offline"

    metrics = load_json(run_dir / "metrics" / "test_metrics_jv.json")
    assert {"precision", "recall", "f1"} <= set(metrics)
    assert set(metrics["per_type"]) == {"PER", "LOC", "ORG"}
    assert "type_confusions" in metrics["error_summary"]

    bootstrap = load_json(run_dir / "metrics" / "bootstrap_jv.json")
    assert bootstrap["n_resamples"] == 100
    assert bootstrap["n_sentences"] == 8

    sentences = load_jsonl(run_dir / "predictions" / "test_jv_sentence_level.jsonl")
    entities = load_jsonl(run_dir / "predictions" / "test_jv_entity_level.jsonl")
    assert len(sentences) == 8
    kinds = {e["kind"] for e in entities}
    assert kinds <= {"tp", "fp", "fn"}

    assert (run_dir / "logs" / "trainer_history.json").exists()
    assert (run_dir / "data" / "truncation_stats.json").exists()
