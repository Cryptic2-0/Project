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

    model_config = {"env_prefix": "MS_", "env_file": ".env"}


settings = Settings()
