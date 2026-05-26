"""OpenTelemetry tracing — optional, enabled when OTEL_EXPORTER_OTLP_ENDPOINT is set.

Local docker-compose runs Tempo at http://tempo:4317 and points OTEL_EXPORTER_OTLP_ENDPOINT
at it. In production, point it at Grafana Cloud OTLP (free tier: 50 GB traces forever) by
setting OTEL_EXPORTER_OTLP_ENDPOINT + OTEL_EXPORTER_OTLP_HEADERS (Basic auth token from
Grafana). If unset, tracing is a no-op — zero cost, no extra dependency at runtime.
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI


def setup_tracing(app: FastAPI) -> None:
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logging.warning("opentelemetry packages not installed; skipping tracing setup")
        return

    resource = Resource.create(
        {
            "service.name": os.environ.get("OTEL_SERVICE_NAME", "moviesentiment"),
            "service.version": os.environ.get("GIT_SHA", "unknown"),
            "deployment.environment": os.environ.get("MS_ENV", "dev"),
        }
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)

    FastAPIInstrumentor.instrument_app(app)
