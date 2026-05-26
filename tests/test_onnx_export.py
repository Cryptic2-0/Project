"""End-to-end test for ONNX export — slow (downloads ~250 MB on first run), marked accordingly.

Skipped by default. Run weekly in CI via:
    pytest -m slow tests/test_onnx_export.py
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.slow
def test_export_quantize_inference_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("optimum")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("torch")

    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # type: ignore[no-untyped-call]

    fp32_dir = tmp_path / "fp32"
    int8_dir = tmp_path / "int8"

    model = ORTModelForSequenceClassification.from_pretrained(model_name, export=True)
    model.save_pretrained(fp32_dir)
    tokenizer.save_pretrained(fp32_dir)
    assert (fp32_dir / "model.onnx").exists()

    quantizer = ORTQuantizer.from_pretrained(fp32_dir)
    qcfg = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=int8_dir, quantization_config=qcfg)
    assert (int8_dir / "model_quantized.onnx").exists() or (int8_dir / "model.onnx").exists()

    import numpy as np
    import onnxruntime as ort

    onnx_file = int8_dir / "model_quantized.onnx"
    if not onnx_file.exists():
        onnx_file = int8_dir / "model.onnx"
    sess = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])

    enc = tokenizer(["a great movie", "terrible film"], return_tensors="np", padding=True)
    logits = sess.run(None, dict(enc))[0]

    assert logits.shape == (2, 2), f"expected (2,2), got {logits.shape}"
    assert np.isfinite(logits).all()
