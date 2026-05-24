"""DistilBERT fine-tuning with HuggingFace Trainer, tracked with MLflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import mlflow.transformers
import numpy as np
import pandas as pd
import yaml
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from moviesentiment.config import settings
from moviesentiment.eval.metrics import log_all_eval


def _softmax(logits: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)  # type: ignore[no-any-return]


def _load_params() -> dict[str, Any]:
    return yaml.safe_load(Path("params.yaml").read_text())["transformer"]  # type: ignore[no-any-return]


def _build_dataset(
    df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase, max_length: int
) -> Dataset:
    ds = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))
    ds = ds.map(
        lambda batch: tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        ),
        batched=True,
        remove_columns=["text"],
    )
    return ds.rename_column("label", "labels")


def _compute_metrics(pred: EvalPrediction) -> dict[str, float]:
    preds = np.argmax(pred.predictions, axis=-1)
    return {
        "accuracy": float(accuracy_score(pred.label_ids, preds)),
        "macro_f1": float(f1_score(pred.label_ids, preds, average="macro")),
    }


def train_transformer() -> None:
    """Fine-tune DistilBERT on train split, evaluate on val, log to MLflow."""
    import torch

    params = _load_params()

    train_df = pd.read_parquet("data/processed/train.parquet")
    val_df = pd.read_parquet("data/processed/val.parquet")

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(params["model_name"])
    train_ds = _build_dataset(train_df, tokenizer, params["max_length"])
    val_ds = _build_dataset(val_df, tokenizer, params["max_length"])

    model = AutoModelForSequenceClassification.from_pretrained(params["model_name"], num_labels=2)

    use_fp16 = torch.cuda.is_available()
    output_dir = str(settings.model_dir / "distilbert-checkpoints")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=params["epochs"],
        per_device_train_batch_size=params["batch_size"],
        per_device_eval_batch_size=params["batch_size"] * 2,
        learning_rate=params["lr"],
        warmup_ratio=params["warmup_ratio"],
        weight_decay=params["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=use_fp16,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute_metrics,
    )

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    with mlflow.start_run(run_name="distilbert-finetune", tags={"model": "distilbert"}) as run:
        mlflow.log_params(params)

        trainer.train()

        val_output = trainer.predict(val_ds)
        logits = np.array(val_output.predictions)
        labels = np.array(val_output.label_ids)
        preds = np.argmax(logits, axis=-1)
        probs = _softmax(logits)[:, 1]

        acc = float(accuracy_score(labels, preds))
        f1 = float(f1_score(labels, preds, average="macro"))
        auc = float(roc_auc_score(labels, probs))

        mlflow.log_metrics({"accuracy": acc, "macro_f1": f1, "roc_auc": auc})
        log_all_eval(pd.Series(labels), preds, probs)

        model_save_path = settings.model_dir / "distilbert"
        trainer.save_model(str(model_save_path))
        tokenizer.save_pretrained(str(model_save_path))

        mlflow.transformers.log_model(
            transformers_model={"model": trainer.model, "tokenizer": tokenizer},
            artifact_path="model",
            task="text-classification",
            registered_model_name=settings.model_name,
        )

        Path("metrics/transformer.json").parent.mkdir(exist_ok=True)
        Path("metrics/transformer.json").write_text(
            json.dumps({"accuracy": acc, "macro_f1": f1, "roc_auc": auc})
        )

        print(f"DistilBERT -- acc={acc:.4f}  f1={f1:.4f}  auc={auc:.4f}")
        print(f"Run ID: {run.info.run_id}")
