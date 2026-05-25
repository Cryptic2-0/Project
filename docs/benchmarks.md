# Inference Benchmarks

Measured on CPU (batch_size=2, max_length=128, n_runs=100).

## Latency

| Model | p50 (ms) | p95 (ms) | p99 (ms) | Mean (ms) | Size (MB) |
|-------|----------|----------|----------|-----------|-----------|
| DistilBERT FP32 ONNX | 14.1 | 16.6 | 17.9 | 14.3 | 256 |
| DistilBERT INT8 ONNX | 6.8 | 30.5 | 52.4 | 9.2 | 64 |

**INT8 speedup: 2.1x** at p50 latency.

## Notes

- ONNX export via `optimum` (path: `models/distilbert_onnx/model.onnx`)
- INT8 dynamic quantization via `onnxruntime.quantization.quantize_dynamic` (QInt8 weights)
- Accuracy degradation from quantization: <1% (see MLflow run comparison)
- Environment: Python 3.10, onnxruntime-cpu, Windows 11
