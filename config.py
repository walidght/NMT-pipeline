from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional
from pathlib import Path


class Settings(BaseSettings):
    model_checkpoint: str = "Helsinki-NLP/opus-mt-en-fr"
    local_model_path: str = str(Path("./results").resolve())
    processed_data_path: str = str(Path("./processed_data").resolve())
    hub_model_id: str = "linguistflow-en-fr-mvp"

    max_length: int = Field(
        default=50, description="Max sequence length for tokenization")
    max_new_tokens: int = Field(
        default=50, description="Max tokens generated during inference")
    beam_size: int = 4

    metric_for_best_model: str = "bleu"
    save_total_limit: int = 2

    wandb_project: str = "linguistflow-nmt"
    wandb_api_key: Optional[str] = None
    hf_token: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False
    )


settings = Settings()
