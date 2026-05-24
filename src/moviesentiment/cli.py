"""CLI entry point — all pipeline commands live here."""

from pathlib import Path

import typer

app = typer.Typer(help="MovieSentiment pipeline CLI")


@app.command()
def scrape(
    out: Path = typer.Option(Path("data/raw/reviews.parquet"), help="Output Parquet path"),
) -> None:
    """Scrape IMDb reviews and save as Parquet. Movie IDs are read from params.yaml."""
    import yaml

    from moviesentiment.data.scrape import scrape_reviews

    ids: list[str] = []
    params_path = Path("params.yaml")
    if params_path.exists():
        with open(params_path) as fh:
            raw = yaml.safe_load(fh)
        if isinstance(raw, dict):
            section = raw.get("scrape") or {}
            if isinstance(section, dict):
                raw_ids = section.get("movie_ids") or []
                if isinstance(raw_ids, list):
                    ids = [str(mid) for mid in raw_ids if mid]

    n = scrape_reviews(movie_ids=ids, out_path=out)
    typer.echo(f"Scraped {n} reviews → {out}")


@app.command()
def clean(
    inp: Path = typer.Option(Path("data/raw/reviews.parquet"), help="Input Parquet"),
    out: Path = typer.Option(Path("data/interim/clean.parquet"), help="Output Parquet"),
) -> None:
    """Clean raw reviews."""
    from moviesentiment.data.clean import clean_reviews

    n = clean_reviews(inp, out)
    typer.echo(f"Cleaned {n} reviews → {out}")


@app.command()
def split(
    inp: Path = typer.Option(Path("data/interim/clean.parquet"), help="Input Parquet"),
    out_dir: Path = typer.Option(Path("data/processed"), help="Output directory"),
) -> None:
    """Split cleaned data into train/val/test."""
    from moviesentiment.data.split import split_dataset

    counts = split_dataset(inp, out_dir)
    typer.echo(f"Split → {counts}")


@app.command()
def train(
    model: str = typer.Argument(..., help="baseline | transformer"),
) -> None:
    """Train a model and log to MLflow."""
    if model == "baseline":
        from moviesentiment.models.baseline import train_baseline

        train_baseline()
    elif model == "transformer":
        from moviesentiment.models.transformer import train_transformer

        train_transformer()
    else:
        typer.echo(f"Unknown model: {model}", err=True)
        raise typer.Exit(1)


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


if __name__ == "__main__":
    app()
