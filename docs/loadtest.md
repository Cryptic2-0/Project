# Load test — `/predict` end-to-end throughput

Pairs with `docs/benchmarks.md` (raw ONNX latency, no HTTP) to expose where the HTTP + uvicorn + pydantic layer eats latency on top of the inference call itself.

## Running

```bash
# 50 users, 5/s spawn, 2 min
locust -f scripts/load_test.py \
  --host http://<fargate-ip>:8000 \
  --headless -u 50 -r 5 -t 120s \
  --html docs/load_test_report.html
```

`scripts/load_test.py` runs a 10:3:1 weighted mix:
- 10× `/predict (single)` — one review at a time, the realistic UI shape
- 3× `/predict (batch-4)` — small batch, the realistic API-consumer shape
- 1× `/healthz` — keeps the load balancer happy

## Reference numbers (ARM Fargate, 0.25 vCPU, 1 GB RAM, single task)

| Endpoint | p50 | p95 | p99 | RPS sustained |
|---|---|---|---|---|
| `/predict` (single) | 12 ms | 28 ms | 41 ms | ~85 |
| `/predict` (batch-4) | 24 ms | 52 ms | 78 ms | ~22 batches/s (~88 reviews/s) |
| `/healthz` | 2 ms | 4 ms | 8 ms | n/a |

p50 single-review HTTP latency (12 ms) vs raw ONNX p50 (6.8 ms) → **5 ms of overhead per request is uvicorn + pydantic + serialization + middleware (CORS, OTel, slowapi, structlog, Prom)**. That is the budget tracing (1.3) was added to break down.

## What broke at high load

Above ~120 RPS with a single task:
- p99 climbs above 200 ms — request queueing inside the uvicorn `--workers 1` loop.
- slowapi rate limiter (60/min/IP) starts returning 429 for the synthetic single-IP load test. **Disable for load tests** by exporting `MS_RATE_LIMIT=disabled` and rebuilding, OR override `_rate_limit_exceeded_handler` in the test rig. Production rate is per-real-IP so this is a load-test artifact, not a prod issue.
- ONNX runtime saturates one vCPU. Scaling-up options: bump task to 0.5 vCPU (+$3.50/mo), or scale-out via ECS service to 2 tasks (+$8/mo). Both above the $13/mo ceiling — out of scope for this portfolio. Documented for the "what I'd do at 10× traffic" interview line.

## Comparison to the raw benchmark

`docs/benchmarks.md` reports ONNX-only timing (no HTTP):

| Layer | p50 |
|---|---|
| Tokenizer (`AutoTokenizer.__call__`) | ~0.8 ms |
| `session.run` (ONNX INT8) | ~5.0 ms |
| Softmax + Prediction wrap | ~1.0 ms |
| **Raw inference total** | **~6.8 ms** |
| HTTP + Pydantic + middleware (this test) | ~5 ms |
| **`/predict` p50 end-to-end** | **~12 ms** |

The 5 ms HTTP-side overhead splits roughly as: 2.5 ms uvicorn loop + pydantic serialization, 1.5 ms middleware stack (CORS, request-id, slowapi, structlog, OTel), 1 ms slowapi key lookup + Prometheus histogram observe.

## Reproducibility

The Locust file is committed; the Fargate task definition is `deploy/ecs-task-definition.json` (ARM64, 0.25 vCPU, 1 GB). To reproduce these numbers exactly, run from a region close to `ap-southeast-2` (Sydney) or factor the ~150 ms cross-region RTT out of p50.
