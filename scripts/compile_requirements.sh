#!/usr/bin/env bash
# Compile pinned, transitively-resolved lockfiles for reproducible installs.
# Run after editing pyproject.toml dependencies. CI installs from these locks.
#
# Output:
#   requirements.lock         runtime only
#   requirements-dev.lock     runtime + dev
#   requirements-train.lock   runtime + train (peft, accelerate, sagemaker, boto3)
#   requirements-monitor.lock runtime + monitor (nannyml, great-expectations)
#   requirements-explain.lock runtime + explain (captum)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

command -v uv >/dev/null 2>&1 || { echo "install uv first: pip install uv"; exit 1; }

echo "compiling runtime lock..."
uv pip compile pyproject.toml -o requirements.lock --quiet

echo "compiling dev lock..."
uv pip compile pyproject.toml --extra dev -o requirements-dev.lock --quiet

echo "compiling train lock..."
uv pip compile pyproject.toml --extra train -o requirements-train.lock --quiet

echo "compiling monitor lock..."
uv pip compile pyproject.toml --extra monitor -o requirements-monitor.lock --quiet

echo "compiling explain lock..."
uv pip compile pyproject.toml --extra explain -o requirements-explain.lock --quiet

echo "done. $(grep -c '^[^#]' requirements.lock 2>/dev/null || echo 0) runtime pins."
