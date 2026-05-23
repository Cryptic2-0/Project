"""Custom Prometheus metrics for the inference service."""

from prometheus_client import Counter, Gauge, Histogram

prediction_confidence = Histogram(
    "prediction_confidence",
    "Per-prediction confidence score",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)

prediction_class_total = Counter(
    "prediction_class_total",
    "Total predictions by label",
    labelnames=["label"],
)

model_version_info = Gauge(
    "model_version_info",
    "Running model metadata",
    labelnames=["model_name", "version", "git_sha"],
)
