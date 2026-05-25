# Inference Benchmarks

Run `make export-onnx` to generate ONNX models and populate this file with real numbers.

## Latency

| Model | p50 (ms) | p95 (ms) | p99 (ms) | Mean (ms) | Size (MB) |
|-------|----------|----------|----------|-----------|-----------|
| DistilBERT FP32 ONNX | TBD | TBD | TBD | TBD | TBD |
| DistilBERT INT8 ONNX | TBD | TBD | TBD | TBD | TBD |

## Notes

- Measured on CPU, batch_size=2, max_length=128, n_runs=100
- INT8 dynamic quantization via `onnxruntime.quantization.quantize_dynamic` (QInt8 weights)
- Run `make export-onnx` to regenerate
