"""Typer CLI smoke tests — covers the thin wrapping layer for each command.

These tests exercise the command-routing logic via `typer.testing.CliRunner`
without invoking the heavy downstream work (training, scraping, ONNX export).
Each downstream function is monkeypatched to a stub so the test stays fast.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from moviesentiment.cli import app

runner = CliRunner()


def test_metrics_missing_dir_returns_exit_1(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    r = runner.invoke(app, ["metrics", "--metrics-dir", str(missing)])
    assert r.exit_code == 1


def test_metrics_renders_existing_jsons(tmp_path: Path) -> None:
    (tmp_path / "foo.json").write_text(json.dumps({"acc": 0.9123, "n": 1000}))
    (tmp_path / "bad.json").write_text("not-json")  # JSONDecodeError branch
    (tmp_path / "list.json").write_text(json.dumps([1, 2, 3]))  # non-dict branch
    r = runner.invoke(app, ["metrics", "--metrics-dir", str(tmp_path)])
    assert r.exit_code == 0
    assert "acc" in r.stdout
    assert "0.9123" in r.stdout


def test_train_unknown_model_exits_1() -> None:
    r = runner.invoke(app, ["train", "no-such-model"])
    assert r.exit_code == 1


def test_train_baseline_invokes_baseline_module(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"baseline": False, "transformer": False, "multitask": False}

    def _stub_baseline() -> None:
        called["baseline"] = True

    import moviesentiment.models.baseline as b

    monkeypatch.setattr(b, "train_baseline", _stub_baseline)
    r = runner.invoke(app, ["train", "baseline"])
    assert r.exit_code == 0
    assert called["baseline"]


def test_train_transformer_invokes_transformer(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, bool] = {"t": False}

    def _stub() -> None:
        called["t"] = True

    import moviesentiment.models.transformer as t

    monkeypatch.setattr(t, "train_transformer", _stub)
    r = runner.invoke(app, ["train", "transformer"])
    assert r.exit_code == 0
    assert called["t"]


def test_train_multitask_invokes_multitask(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, bool] = {"m": False}

    def _stub() -> None:
        called["m"] = True

    import moviesentiment.models.multitask_train as mt

    monkeypatch.setattr(mt, "train_multitask", _stub)
    r = runner.invoke(app, ["train", "multitask"])
    assert r.exit_code == 0
    assert called["m"]


def test_scrape_command_routes_to_scrape_reviews(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _stub(movie_ids: list[str], out_path: Path, source: str) -> int:
        captured["source"] = source
        captured["out"] = out_path
        captured["ids"] = movie_ids
        return 7

    import moviesentiment.data.scrape as sc

    monkeypatch.setattr(sc, "scrape_reviews", _stub)
    r = runner.invoke(app, ["scrape", "--out", "data/raw/test.parquet"])
    assert r.exit_code == 0
    assert captured["source"] == "hf"
    assert "Scraped 7" in r.stdout


def test_clean_command_routes_to_clean_reviews(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inp = tmp_path / "raw.parquet"
    out = tmp_path / "clean.parquet"
    inp.write_text("")  # path must exist for click.Path validation

    import moviesentiment.data.clean as cl

    monkeypatch.setattr(cl, "clean_reviews", lambda i, o: 42)
    r = runner.invoke(app, ["clean", "--inp", str(inp), "--out", str(out)])
    assert r.exit_code == 0
    assert "Cleaned 42" in r.stdout


def test_split_command_routes_to_split_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inp = tmp_path / "clean.parquet"
    inp.write_text("")

    import moviesentiment.data.split as sp

    monkeypatch.setattr(sp, "split_dataset", lambda i, o: {"train": 5, "val": 2, "test": 2})
    r = runner.invoke(app, ["split", "--inp", str(inp), "--out-dir", str(tmp_path)])
    assert r.exit_code == 0
    assert "train" in r.stdout


def test_export_onnx_command_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, bool] = {"x": False}

    def _stub() -> None:
        called["x"] = True

    import moviesentiment.models.onnx_export as oe

    monkeypatch.setattr(oe, "main", _stub)
    r = runner.invoke(app, ["export-onnx"])
    assert r.exit_code == 0
    assert called["x"]


def test_drift_command_routes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import moviesentiment.monitor.drift as dr

    monkeypatch.setattr(dr, "run_drift_report", lambda r, c, o: tmp_path / "x.html")
    r = runner.invoke(app, ["drift"])
    assert r.exit_code == 0
    assert "x.html" in r.stdout


def test_prep_emotion_command_routes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import moviesentiment.data.multitask_loaders as ml

    monkeypatch.setattr(ml, "load_emotion", lambda out: out)
    r = runner.invoke(app, ["prep-emotion", "--out", str(tmp_path / "e.parquet")])
    assert r.exit_code == 0


def test_prep_helpfulness_command_routes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import moviesentiment.data.multitask_loaders as ml

    reviews = tmp_path / "r.parquet"
    reviews.write_text("")
    monkeypatch.setattr(ml, "load_helpfulness", lambda r, out: out)
    r = runner.invoke(
        app,
        ["prep-helpfulness", "--reviews", str(reviews), "--out", str(tmp_path / "h.parquet")],
    )
    assert r.exit_code == 0


def test_prep_spoiler_command_routes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import moviesentiment.data.multitask_loaders as ml

    csv = tmp_path / "raw.csv"
    csv.write_text("text,is_spoiler\nhello,0\n")
    monkeypatch.setattr(ml, "load_spoiler", lambda p, out: out)
    r = runner.invoke(app, ["prep-spoiler", str(csv), "--out", str(tmp_path / "s.parquet")])
    assert r.exit_code == 0


def test_insights_batch_command_routes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import moviesentiment.serve.insights as ins

    monkeypatch.setattr(ins, "materialise_all", lambda out: 3)
    r = runner.invoke(app, ["insights-batch", "--out", str(tmp_path)])
    assert r.exit_code == 0
    assert "3" in r.stdout
