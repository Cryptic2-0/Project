"""Calibration regression gate — covers the Brier score helper + CI hook."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CHECK_SCRIPT = _REPO_ROOT / "scripts" / "check_calibration.py"

# Skip MLflow side effects: compute_and_log_brier writes to disk + logs to
# MLflow. Patching mlflow is cheaper than running a tracking server.


def test_brier_perfect_predictions_returns_zero(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    import mlflow

    monkeypatch.setattr(mlflow, "log_metric", lambda *a, **k: None)

    from moviesentiment.eval.metrics import compute_and_log_brier

    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.0, 1.0, 0.0, 1.0])
    score = compute_and_log_brier(y_true, y_prob)
    assert abs(score) < 1e-9

    data = json.loads((tmp_path / "metrics" / "brier.json").read_text())
    assert data["brier"] == score
    assert data["threshold"] > 0


def test_brier_worst_case_predictions_returns_one(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    import mlflow

    monkeypatch.setattr(mlflow, "log_metric", lambda *a, **k: None)

    from moviesentiment.eval.metrics import compute_and_log_brier

    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([1.0, 0.0, 1.0, 0.0])
    score = compute_and_log_brier(y_true, y_prob)
    assert abs(score - 1.0) < 1e-9


def test_check_calibration_passes_on_good_score(tmp_path: Path) -> None:
    (tmp_path / "metrics").mkdir()
    (tmp_path / "metrics" / "brier.json").write_text(json.dumps({"brier": 0.05, "threshold": 0.10}))
    result = subprocess.run(
        [sys.executable, str(_CHECK_SCRIPT), "--path", "metrics/brier.json"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_check_calibration_fails_on_regression(tmp_path: Path) -> None:
    (tmp_path / "metrics").mkdir()
    (tmp_path / "metrics" / "brier.json").write_text(json.dumps({"brier": 0.25, "threshold": 0.10}))
    result = subprocess.run(
        [sys.executable, str(_CHECK_SCRIPT), "--path", "metrics/brier.json"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "FAIL" in result.stderr


def test_check_calibration_allow_missing(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(_CHECK_SCRIPT),
            "--path",
            str(tmp_path / "nope.json"),
            "--allow-missing",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
