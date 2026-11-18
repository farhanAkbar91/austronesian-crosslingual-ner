"""Training orchestration for token classification via Hugging Face Trainer.

Design notes:
  * A custom padding collator is used so the same code path works with real
    tokenizers and the offline stub tokenizer from ``axner.models.offline``.
  * ``remove_unused_columns=False`` keeps export columns (tokens, kept_word_ids,
    sentence_id, lang) available for prediction export; the collator only
    forwards model inputs.
  * Checkpoint selection restores the best epoch by the configured validation
    metric (``training.checkpoint.selection_metric``, e.g. ``entity_f1``).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from axner.constants import ENTITY_TYPES, IGNORE_INDEX
from axner.utils.logging import get_logger
from axner.utils.seed import set_deterministic

logger = get_logger("axner.models.training")


class TokenCollator:
    """Pad ``input_ids``/``attention_mask``/``labels`` into torch tensors.

    Non-model columns are dropped here (not via ``remove_unused_columns``)
    so datasets can carry export metadata through prediction.
    """

    def __init__(self, pad_token_id: int = 0, label_pad_token_id: int = IGNORE_INDEX):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features):
        import torch

        max_len = max(len(f["input_ids"]) for f in features)

        def pad(values, pad_value):
            return [list(v) + [pad_value] * (max_len - len(v)) for v in values]

        return {
            "input_ids": torch.tensor(
                pad([f["input_ids"] for f in features], self.pad_token_id), dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                pad([f["attention_mask"] for f in features], 0), dtype=torch.long
            ),
            "labels": torch.tensor(
                pad([f["labels"] for f in features], self.label_pad_token_id), dtype=torch.long
            ),
        }


def build_compute_metrics(label_names):
    """Build a ``compute_metrics`` callable computing entity-level metrics.

    Returned keys are prefixed ``eval_`` by the Trainer; the F1 key
    ``entity_f1`` drives best-checkpoint selection.
    """

    from axner.evaluation.entity_metrics import compute_entity_metrics, per_type_metrics

    def compute_metrics(eval_pred):
        predictions = eval_pred.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        label_ids = eval_pred.label_ids
        pred_ids = np.argmax(predictions, axis=-1)
        mask = label_ids != IGNORE_INDEX

        gold_seqs = [
            [label_names[t] for t, keep in zip(lr, mr) if keep]
            for lr, mr in zip(label_ids, mask)
        ]
        pred_seqs = [
            [label_names[t] for t, keep in zip(pr, mr) if keep]
            for pr, mr in zip(pred_ids, mask)
        ]

        micro = compute_entity_metrics(gold_seqs, pred_seqs)
        per_type = per_type_metrics(gold_seqs, pred_seqs, ENTITY_TYPES)
        out = {
            "entity_precision": micro["precision"],
            "entity_recall": micro["recall"],
            "entity_f1": micro["f1"],
            "entity_tp": micro["tp"],
            "entity_fp": micro["fp"],
            "entity_fn": micro["fn"],
            "n_sentences": micro["n_sentences"],
        }
        for etype, entry in per_type.items():
            out[f"{etype.lower()}_f1"] = entry["f1"]
        return out

    return compute_metrics


def _training_arguments(output_dir, seed, training_cfg, device, eval_only=False):
    """Build TrainingArguments with cross-version compatibility."""
    from transformers import TrainingArguments

    hp = training_cfg["hyperparameters"]
    field_names = {f.name for f in dataclasses.fields(TrainingArguments)}
    eval_strategy_key = "eval_strategy" if "eval_strategy" in field_names else "evaluation_strategy"

    kwargs = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=hp["batch_size"],
        per_device_eval_batch_size=training_cfg.get("eval_batch_size", 32),
        learning_rate=hp["learning_rate"],
        num_train_epochs=hp["epochs"],
        weight_decay=hp["weight_decay"],
        warmup_ratio=hp["warmup_ratio"],
        max_grad_norm=hp["max_grad_norm"],
        logging_steps=training_cfg.get("logging_steps", 50),
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=training_cfg.get("checkpoint", {}).get("selection_metric", "entity_f1"),
        greater_is_better=True,
        save_total_limit=training_cfg.get("checkpoint", {}).get("save_total_limit", 2),
        seed=seed,
        data_seed=seed,
        fp16=bool(training_cfg.get("fp16", False)) and device == "cuda",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        use_cpu=device == "cpu",
    )
    # eval_strategy was renamed from evaluation_strategy across versions.
    kwargs[eval_strategy_key] = kwargs.pop("eval_strategy")
    if eval_only:
        kwargs["per_device_train_batch_size"] = kwargs["per_device_eval_batch_size"]
        kwargs["num_train_epochs"] = 1
        kwargs["eval_strategy"] = "no"
        kwargs["save_strategy"] = "no"
        kwargs["load_best_model_at_end"] = False
    return TrainingArguments(**kwargs)


def _make_trainer(model, tokenizer, args, train_ds, val_ds, pad_token_id, compute_metrics):
    from transformers import Trainer

    import inspect

    kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=TokenCollator(pad_token_id=pad_token_id),
        compute_metrics=compute_metrics,
    )
    if tokenizer is not None:
        param = "processing_class" if "processing_class" in inspect.signature(Trainer.__init__).parameters else "tokenizer"
        kwargs[param] = tokenizer
    return Trainer(**kwargs)


def run_training(model, tokenizer, train_ds, val_ds, output_dir, seed,
                 training_cfg, compute_metrics, pad_token_id: int = 0):
    """Train a token classifier; returns the Trainer (best model restored).

    ``compute_metrics`` must be provided (use :func:`build_compute_metrics`);
    it drives validation-based checkpoint selection.
    """
    device = resolve_device_pref(training_cfg)
    set_deterministic(seed)
    args = _training_arguments(output_dir, seed, training_cfg, device)
    trainer = _make_trainer(model, tokenizer, args, train_ds, val_ds, pad_token_id, compute_metrics)
    logger.info(
        "training on device=%s seed=%d train_rows=%d val_rows=%d epochs=%s batch=%s lr=%s",
        device, seed, train_ds.num_rows, val_ds.num_rows,
        training_cfg["hyperparameters"]["epochs"], training_cfg["hyperparameters"]["batch_size"],
        training_cfg["hyperparameters"]["learning_rate"],
    )
    trainer.train()
    logger.info("best checkpoint: %s", trainer.state.best_model_checkpoint)
    return trainer


def resolve_device_pref(training_cfg) -> str:
    from axner.models.init import resolve_device

    return resolve_device(training_cfg.get("device", "auto"))


def make_eval_trainer(model, tokenizer, eval_ds, training_cfg, compute_metrics, pad_token_id: int = 0):
    """Build a Trainer for evaluation/prediction only (no training)."""
    device = resolve_device_pref(training_cfg)
    args = _training_arguments(
        output_dir=training_cfg.get("_eval_output_dir", "outputs/_eval_tmp"),
        seed=training_cfg.get("_seed", 0),
        training_cfg=training_cfg,
        device=device,
        eval_only=True,
    )
    return _make_trainer(model, tokenizer, args, None, eval_ds, pad_token_id, compute_metrics)


def predict_dataset(trainer, ds):
    """Run prediction; returns (predictions, label_ids, metrics)."""
    output = trainer.predict(ds)
    return output.predictions, output.label_ids, output.metrics


def trainer_log_history(trainer) -> list:
    return list(trainer.state.log_history)


def best_checkpoint_path(trainer) -> str | None:
    return trainer.state.best_model_checkpoint
