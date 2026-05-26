from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data"
    model_dir: Path = project_root / "models"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment: str = "moviesentiment"
    model_name: str = "moviesentiment-classifier"
    model_stage: str = "Production"
    max_text_length: int = 5000
    max_batch_size: int = 32

    # Comma-separated list of allowed CORS origins. Default `*` for the demo / public
    # API; tighten in production by setting MS_CORS_ALLOW_ORIGINS to e.g.
    # "https://cryptic2-0.github.io,http://localhost:3000".
    cors_allow_origins: str = "*"

    # Pin a HuggingFace model revision (commit SHA or tag) to defend against namespace
    # takeover. Default empty = main; set MS_HF_REVISION in CI to the resolved commit.
    hf_revision: str = ""

    model_config = {"env_prefix": "MS_", "env_file": ".env"}

    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]


settings = Settings()
