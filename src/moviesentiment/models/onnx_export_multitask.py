"""Export the multi-task DistilBERT checkpoint to ONNX with five named outputs.

Five named outputs is the contract `MultiTaskInferenceEngine.from_disk` enforces:
    sentiment    : (B, 2)   float32
    aspect       : (B, 5, 3) float32
    emotion      : (B, 6)    float32
    spoiler      : (B, 2)    float32
    helpfulness  : (B,)      float32

Quantisation: dynamic INT8 weights via onnxruntime.quantization. The benchmark
step is inherited from the v1 export script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from moviesentiment.config import settings


def export_multitask_onnx(
    checkpoint_dir: Path,
    out_dir: Path | None = None,
    opset: int = 14,
    max_length: int = 256,
) -> Path:
    """Trace the trained checkpoint into ONNX. Returns path to model.onnx (INT8)."""
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoTokenizer

    from moviesentiment.models.multitask import build_multitask_model

    out = out_dir or settings.model_dir / "distilbert_multitask_onnx"
    out.mkdir(parents=True, exist_ok=True)

    # nosec B615 - checkpoint_dir is a local DVC-pulled path, not a Hub identifier.
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))  # nosec B615
    tokenizer.save_pretrained(str(out))

    # Rebuild architecture, load weights. We don't store the model class on disk —
    # only the encoder name + head sizes — so we reconstruct from params.
    import yaml

    params = yaml.safe_load(Path("params.yaml").read_text()).get("multitask", {})
    model_name = params.get("model_name", "distilbert-base-uncased")
    revision = params.get("revision") or settings.hf_revision or None
    model = build_multitask_model(model_name, revision=revision)
    # nosec B614 - checkpoint is produced by our own SageMaker job and stored in DVC.
    state = torch.load(
        checkpoint_dir / "model.pt", map_location="cpu", weights_only=True
    )  # nosec B614
    model.load_state_dict(state)
    model.eval()

    # Wrap so torch.onnx.export sees a tuple of named tensors, not the dataclass.
    class _OnnxWrap(torch.nn.Module):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, input_ids: Any, attention_mask: Any) -> tuple[Any, ...]:
            out = self.inner(input_ids=input_ids, attention_mask=attention_mask)
            return (
                out.sentiment_logits,
                out.aspect_logits,
                out.emotion_logits,
                out.spoiler_logits,
                out.helpfulness,
            )

    wrap = _OnnxWrap(model).eval()

    dummy = tokenizer(
        ["A wonderful film about hope."] * 2,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    fp32_path = out / "model_fp32.onnx"
    # Legacy TorchScript exporter (dynamo=False). The new dynamo exporter
    # emits external-data files and breaks onnxruntime.quantize_dynamic with
    # a shape-inference error on multi-output graphs (helpfulness collapses
    # to (B,) after squeeze). The legacy exporter is sufficient here.
    torch.onnx.export(
        wrap,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["sentiment", "aspect", "emotion", "spoiler", "helpfulness"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "sentiment": {0: "batch"},
            "aspect": {0: "batch"},
            "emotion": {0: "batch"},
            "spoiler": {0: "batch"},
            "helpfulness": {0: "batch"},
        },
        opset_version=opset,
        dynamo=False,
    )

    int8_path = out / "model.onnx"
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
    return int8_path


def main() -> None:
    ckpt = settings.model_dir / "distilbert_multitask"
    if not (ckpt / "model.pt").exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt}. Run `moviesentiment train multitask` first."
        )
    path = export_multitask_onnx(ckpt)
    print(f"Multi-task ONNX INT8 -> {path}")


if __name__ == "__main__":
    main()
