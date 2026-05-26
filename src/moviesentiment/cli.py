"""CLI entry point — all pipeline commands live here."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="MovieSentiment pipeline CLI")
_console = Console()


@app.command()
def scrape(
    out: Path = typer.Option(Path("data/raw/reviews.parquet"), help="Output Parquet path"),
) -> None:
    """Scrape IMDb reviews and save as Parquet. Movie IDs are read from params.yaml."""
    import yaml

    from moviesentiment.data.scrape import scrape_reviews

    ids: list[str] = []
    source = "hf"
    params_path = Path("params.yaml")
    if params_path.exists():
        with open(params_path) as fh:
            raw = yaml.safe_load(fh)
        if isinstance(raw, dict):
            section = raw.get("scrape") or {}
            if isinstance(section, dict):
                source = str(section.get("source") or "hf")
                raw_ids = section.get("movie_ids") or []
                if isinstance(raw_ids, list):
                    ids = [str(mid) for mid in raw_ids if mid]

    n = scrape_reviews(movie_ids=ids, out_path=out, source=source)
    typer.echo(f"Scraped {n} reviews -> {out}")


@app.command()
def clean(
    inp: Path = typer.Option(Path("data/raw/reviews.parquet"), help="Input Parquet"),
    out: Path = typer.Option(Path("data/interim/clean.parquet"), help="Output Parquet"),
) -> None:
    """Clean raw reviews."""
    from moviesentiment.data.clean import clean_reviews

    n = clean_reviews(inp, out)
    typer.echo(f"Cleaned {n} reviews -> {out}")


@app.command()
def split(
    inp: Path = typer.Option(Path("data/interim/clean.parquet"), help="Input Parquet"),
    out_dir: Path = typer.Option(Path("data/processed"), help="Output directory"),
) -> None:
    """Split cleaned data into train/val/test."""
    from moviesentiment.data.split import split_dataset

    counts = split_dataset(inp, out_dir)
    typer.echo(f"Split -> {counts}")


@app.command()
def train(
    model: str = typer.Argument(..., help="baseline | transformer | multitask"),
) -> None:
    """Train a model and log to MLflow."""
    if model == "baseline":
        from moviesentiment.models.baseline import train_baseline

        train_baseline()
    elif model == "transformer":
        from moviesentiment.models.transformer import train_transformer

        train_transformer()
    elif model == "multitask":
        from moviesentiment.models.multitask_train import train_multitask

        train_multitask()
    else:
        typer.echo(f"Unknown model: {model}", err=True)
        raise typer.Exit(1)


@app.command(name="insights-batch")
def insights_batch(
    out: Path = typer.Option(Path("data/production/insights"), help="Output dir"),
) -> None:
    """Materialise per-movie insights JSON over the reservoir sample."""
    from moviesentiment.serve.insights import materialise_all

    n = materialise_all(out)
    typer.echo(f"Wrote insights for {n} movie(s) -> {out}")


@app.command()
def serve_api(
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(8000),
) -> None:
    """Start the FastAPI inference server."""
    import uvicorn

    uvicorn.run("moviesentiment.serve.api:app", host=host, port=port, reload=False)


@app.command()
def drift(
    reference: Path = typer.Option(Path("data/processed/train.parquet")),
    current: Path = typer.Option(Path("data/production/recent.parquet")),
    out: Path = typer.Option(Path("docs/drift_reports")),
) -> None:
    """Run Evidently drift detection."""
    from moviesentiment.monitor.drift import run_drift_report

    result = run_drift_report(reference, current, out)
    typer.echo(f"Drift report: {result}")


@app.command()
def export_onnx() -> None:
    """Export DistilBERT to ONNX (FP32 + INT8) and benchmark latency."""
    from moviesentiment.models.onnx_export import main as _export

    _export()


@app.command()
def metrics(
    metrics_dir: Path = typer.Option(Path("metrics"), help="Directory of metric JSON files"),
) -> None:
    """Render all metric JSON files as a Rich table."""
    import json

    table = Table(title="MovieSentiment metrics", show_header=True, header_style="bold")
    table.add_column("file", style="dim")
    table.add_column("metric")
    table.add_column("value", justify="right")

    if not metrics_dir.exists():
        _console.print(f"[red]No metrics directory at {metrics_dir}[/red]")
        raise typer.Exit(1)

    for path in sorted(metrics_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        for key, value in data.items() if isinstance(data, dict) else []:
            table.add_row(
                path.name, str(key), f"{value:.4f}" if isinstance(value, float) else str(value)
            )

    _console.print(table)


@app.command(name="perf-estimate")
def perf_estimate(
    reference: Path = typer.Option(Path("data/production/labeled_reference.parquet")),
    current: Path = typer.Option(Path("data/production/recent.parquet")),
) -> None:
    """Run NannyML CBPE to estimate production F1 without ground truth."""
    from moviesentiment.monitor.perf_estimate import estimate

    result = estimate(reference, current)
    table = Table(title="Estimated production performance")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for k, v in result.items():
        table.add_row(k, f"{v:.4f}")
    _console.print(table)


if __name__ == "__main__":
    app()
