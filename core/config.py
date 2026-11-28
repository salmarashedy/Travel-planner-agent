from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = Field(default="Travel Agent", validation_alias="APP_NAME")
    app_version: str = Field(default="0.1.0", validation_alias="APP_VERSION")
    ollama_base_url: str = Field(
        default="http://localhost:11434", validation_alias="OLLAMA_BASE_URL"
    )
    ollama_model: str = Field(default="qwen3:4b", validation_alias="OLLAMA_MODEL")
    ollama_timeout: float = Field(default=60.0, gt=0, validation_alias="OLLAMA_TIMEOUT")
    ollama_temperature: float = Field(
        default=0.0, ge=0, validation_alias="OLLAMA_TEMPERATURE"
    )
    ollama_seed: int = Field(default=42, validation_alias="OLLAMA_SEED")
    ollama_num_ctx: int = Field(default=8192, gt=0, validation_alias="OLLAMA_NUM_CTX")
    database_url: str | None = Field(default=None, validation_alias="DATABASE_URL")
    mlflow_tracking_uri: str = Field(
        # MLflow 3 refuses a plain ./mlruns folder
        default="sqlite:///mlflow.db", validation_alias="MLFLOW_TRACKING_URI"
    )
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")


@lru_cache
def get_settings() -> Settings:
    return Settings()
