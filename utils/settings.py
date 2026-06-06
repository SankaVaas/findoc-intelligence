"""
Central settings — loaded once at startup from .env / environment variables.
All other modules import `settings` from here; never read os.environ directly.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "mistral:7b-instruct-q4_K_M"

    # ── Embeddings ───────────────────────────────────────────
    embed_model: str = "intfloat/multilingual-e5-small"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embed_device: str = "cpu"

    # ── Vector store ─────────────────────────────────────────
    chroma_persist_dir: Path = Path("./data/embeddings")
    chroma_collection: str = "findoc"

    # ── Database ─────────────────────────────────────────────
    sqlite_path: Path = Path("./data/processed/findoc.db")

    # ── Data paths ───────────────────────────────────────────
    raw_docs_dir: Path = Path("./data/raw")
    processed_dir: Path = Path("./data/processed")

    # ── API ──────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    log_level: str = "INFO"

    # ── Langfuse ─────────────────────────────────────────────
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # ── Prometheus ───────────────────────────────────────────
    prometheus_port: int = 9090

    # ── Voice ────────────────────────────────────────────────
    whisper_model: str = "medium"
    tts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    voice_device: str = "cpu"

    @property
    def tracing_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    def ensure_dirs(self) -> None:
        """Create all data directories if they don't exist."""
        for path in [
            self.chroma_persist_dir,
            self.sqlite_path.parent,
            self.raw_docs_dir,
            self.processed_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
