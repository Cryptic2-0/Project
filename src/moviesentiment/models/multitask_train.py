"""Joint training entrypoint for the multi-task DistilBERT.

Loads task-specific parquets, concatenates them into a single dataset where
each row is supervised on a subset of tasks (missing labels are -100 / NaN),
and runs a standard HuggingFace `Trainer` loop with the model's
`MultiTaskOutput.loss`.

Inputs (parquet) — each optional, paths read from `params.yaml::multitask`:

    sentiment    : columns = text, label  (0/1)
    spoiler      : columns = text, label  (0/1)
    emotion      : columns = text, label  (0..5 by EMOTIONS index)
    absa         : columns = text, label  (length-5 list, -100|0|1|2 per slot)
    helpfulness  : columns = text, label  (float in [0,1] or NaN)

The script lives outside `cli.py` because torch imports are heavy and we want
the CLI bootstrap to stay quick.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def _load_params() -> dict[str, Any]:
    return yaml.safe_load(Path("params.yaml").read_text()).get("multitask", {})


def _build_combined_dataset(params: dict[str, Any], tokenizer: Any) -> Any:
    """Stitch every task parquet into one in-memory list of token dicts.

    Avoids `datasets.concatenate_datasets` + pyarrow because schema unification
    quietly converts our `-100` / `NaN` sentinels into pyarrow `null`, which
    surfaces in the DataLoader collate as `Could not infer dtype of NoneType`.
    A plain Python list preserves the exact Python ints / floats we wrote.
    """
    import random

    import pandas as pd
    from torch.utils.data import Dataset

    max_length = int(params.get("max_length", 256))
    seed = int(params.get("seed", 42))

    def _row(text: str, **labels: Any) -> dict[str, Any]:
        enc = tokenizer(
            str(text),
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        # Sentinel defaults; per-task block overrides only the label this task
        # supervises. Plain Python ints / floats — never None.
        defaults: dict[str, Any] = {
            "labels_sentiment": -100,
            "labels_emotion": -100,
            "labels_spoiler": -100,
            "labels_aspect": [-100, -100, -100, -100, -100],
            "labels_helpfulness": float("nan"),
        }
        defaults.update(labels)
        return {**enc, **defaults}

    rows: list[dict[str, Any]] = []

    def _ingest(p: str, label_key: str, cast: Any = int) -> int:
        df = pd.read_parquet(p)
        before = len(rows)
        for text, label in zip(df["text"].tolist(), df["label"].tolist(), strict=False):
            if text is None:
                continue
            value = cast(label) if label is not None else None
            if value is None:
                continue
            rows.append(_row(text, **{label_key: value}))
        return len(rows) - before

    counts: dict[str, int] = {}
    if (p := params.get("sentiment_path")) and Path(p).exists():
        counts["sentiment"] = _ingest(p, "labels_sentiment", int)
    if (p := params.get("spoiler_path")) and Path(p).exists():
        counts["spoiler"] = _ingest(p, "labels_spoiler", int)
    if (p := params.get("emotion_path")) and Path(p).exists():
        counts["emotion"] = _ingest(p, "labels_emotion", int)
    if (p := params.get("absa_path")) and Path(p).exists():
        # absa labels are length-5 lists already.
        df = pd.read_parquet(p)
        c = 0
        for text, label in zip(df["text"].tolist(), df["label"].tolist(), strict=False):
            if text is None or label is None:
                continue
            rows.append(_row(text, labels_aspect=[int(x) for x in label]))
            c += 1
        counts["absa"] = c
    if (p := params.get("helpfulness_path")) and Path(p).exists():
        counts["helpfulness"] = _ingest(p, "labels_helpfulness", float)

    if not rows:
        raise FileNotFoundError(
            "No task parquets found. See docs/future_improvements.md for the v2 "
            "training-data plan; populate at least one of "
            "params.yaml::multitask.*_path."
        )

    rng = random.Random(seed)
    rng.shuffle(rows)
    print(f"[multitask] joint dataset n={len(rows)} per-task={counts}")

    class _ListDataset(Dataset[dict[str, Any]]):
        def __init__(self, items: list[dict[str, Any]]) -> None:
            self._items = items

        def __len__(self) -> int:
            return len(self._items)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            return self._items[idx]

    return _ListDataset(rows)


def train_multitask() -> None:
    """Joint fine-tune. Run via `moviesentiment train-multitask`."""
    import mlflow
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    from moviesentiment.config import settings
    from moviesentiment.models.multitask import build_multitask_model

    params = _load_params()
    model_name: str = params.get("model_name", "distilbert-base-uncased")
    revision: str = params.get("revision") or settings.hf_revision or None
    lr = float(params.get("lr", 2e-5))
    epochs = int(params.get("epochs", 2))
    batch_size = int(params.get("batch_size", 16))
    warmup_ratio = float(params.get("warmup_ratio", 0.1))
    weight_decay = float(params.get("weight_decay", 0.01))
    loss_weights: dict[str, float] = params.get("loss_weights", {})

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    ds = _build_combined_dataset(params, tokenizer)

    model = build_multitask_model(model_name, revision=revision)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        from transformers import DataCollatorWithPadding

        pad = DataCollatorWithPadding(tokenizer)
        # Pull label columns out before padding; collator only knows input_ids.
        # datasets / pyarrow stores missing values as Python None, which
        # torch.tensor cannot type-infer ("Could not infer dtype of NoneType").
        # Guard every scalar AND every element of the aspect list.
        label_cols = (
            "labels_sentiment",
            "labels_emotion",
            "labels_spoiler",
            "labels_aspect",
            "labels_helpfulness",
        )

        def _as_int(v: Any, default: int = -100) -> int:
            return default if v is None else int(v)

        def _as_int_list(vs: Any, default: int = -100, n: int = 5) -> list[int]:
            if vs is None:
                return [default] * n
            return [_as_int(x, default) for x in vs]

        def _as_float(v: Any) -> float:
            return float("nan") if v is None else float(v)

        labels = {
            "labels_sentiment": torch.tensor(
                [_as_int(b.get("labels_sentiment")) for b in batch], dtype=torch.long
            ),
            "labels_emotion": torch.tensor(
                [_as_int(b.get("labels_emotion")) for b in batch], dtype=torch.long
            ),
            "labels_spoiler": torch.tensor(
                [_as_int(b.get("labels_spoiler")) for b in batch], dtype=torch.long
            ),
            "labels_aspect": torch.tensor(
                [_as_int_list(b.get("labels_aspect")) for b in batch], dtype=torch.long
            ),
            "labels_helpfulness": torch.tensor(
                [_as_float(b.get("labels_helpfulness")) for b in batch], dtype=torch.float32
            ),
        }
        text_batch = [{k: v for k, v in b.items() if k not in label_cols} for b in batch]
        out = pad(text_batch)
        out.update(labels)
        return out

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    total_steps = len(loader) * epochs
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=int(warmup_ratio * total_steps), num_training_steps=total_steps
    )

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    with mlflow.start_run(run_name="multitask-distilbert", tags={"model": "multitask"}):
        mlflow.log_params({**params, "n_examples": len(ds)})
        model.train()
        step = 0
        running = 0.0
        window_steps = 0
        for _epoch in range(epochs):
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optim.zero_grad()
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels_sentiment=batch["labels_sentiment"],
                    labels_aspect=batch["labels_aspect"],
                    labels_emotion=batch["labels_emotion"],
                    labels_spoiler=batch["labels_spoiler"],
                    labels_helpfulness=batch["labels_helpfulness"],
                    loss_weights=loss_weights,
                )
                if out.loss is None:
                    continue
                out.loss.backward()
                optim.step()
                sched.step()
                running += float(out.loss.item())
                step += 1
                window_steps += 1
                if step % 50 == 0:
                    mlflow.log_metric("loss", running / 50, step=step)
                    running = 0.0
                    window_steps = 0

        save_dir = settings.model_dir / "distilbert_multitask"
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")
        tokenizer.save_pretrained(str(save_dir))
        Path("metrics/multitask.json").parent.mkdir(exist_ok=True)
        final_loss_avg = (running / window_steps) if window_steps > 0 else 0.0
        Path("metrics/multitask.json").write_text(
            json.dumps({"steps": step, "loss_avg": final_loss_avg})
        )
        print(f"Saved multi-task checkpoint to {save_dir}")
