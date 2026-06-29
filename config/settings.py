"""
config/settings.py
──────────────────
Central configuration loaded from environment variables / .env file.
All settings are typed, validated, and documented.

Usage:
    from config.settings import get_settings
    settings = get_settings()
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Base paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class AppSettings(BaseSettings):
    """Top-level application identity and runtime mode."""

    name: str = Field("Clinical Psychology Assistant", alias="APP_NAME")
    version: str = Field("0.1.0", alias="APP_VERSION")
    env: Literal["development", "staging", "production"] = Field(
        "development", alias="APP_ENV"
    )
    debug: bool = Field(False, alias="DEBUG")
    secret_key: str = Field(..., alias="SECRET_KEY")
    allowed_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        alias="ALLOWED_ORIGINS",
    )

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_origins(cls, v: str | list) -> list[str]:
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


class ServerSettings(BaseSettings):
    """FastAPI / Uvicorn server configuration."""

    host: str = Field("0.0.0.0", alias="API_HOST")
    port: int = Field(8000, alias="API_PORT")
    workers: int = Field(1, alias="API_WORKERS")
    reload: bool = Field(True, alias="API_RELOAD")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


class LLMSettings(BaseSettings):
    """Ollama / LLM inference configuration."""

    ollama_base_url: str = Field("http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field("llama3.1:8b", alias="OLLAMA_MODEL")
    ollama_timeout: int = Field(120, alias="OLLAMA_TIMEOUT")

    temperature: float = Field(0.1, alias="LLM_TEMPERATURE", ge=0.0, le=2.0)
    top_k: int = Field(40, alias="LLM_TOP_K", ge=1)
    top_p: float = Field(0.9, alias="LLM_TOP_P", ge=0.0, le=1.0)
    max_tokens: int = Field(2048, alias="LLM_MAX_TOKENS", ge=1)
    context_window: int = Field(8192, alias="LLM_CONTEXT_WINDOW", ge=512)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


class EmbeddingSettings(BaseSettings):
    """Sentence-transformer embedding model configuration."""

    model_name: str = Field("BAAI/bge-large-en-v1.5", alias="EMBEDDING_MODEL")
    device: Literal["cpu", "cuda", "mps"] = Field("cpu", alias="EMBEDDING_DEVICE")
    batch_size: int = Field(32, alias="EMBEDDING_BATCH_SIZE", ge=1)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


class ChromaSettings(BaseSettings):
    """ChromaDB vector store configuration."""

    persist_dir: Path = Field(
        PROJECT_ROOT / "data" / "chroma", alias="CHROMA_PERSIST_DIR"
    )
    collection_name: str = Field(
        "clinical_knowledge_base", alias="CHROMA_COLLECTION_NAME"
    )
    distance_function: Literal["cosine", "l2", "ip"] = Field(
        "cosine", alias="CHROMA_DISTANCE_FUNCTION"
    )

    @field_validator("persist_dir", mode="before")
    @classmethod
    def resolve_path(cls, v: str | Path) -> Path:
        p = Path(v)
        p.mkdir(parents=True, exist_ok=True)
        return p

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


class RAGSettings(BaseSettings):
    """Retrieval-Augmented Generation pipeline configuration."""

    top_k: int = Field(10, alias="RAG_TOP_K", ge=1, le=50)
    rerank_top_k: int = Field(4, alias="RAG_RERANK_TOP_K", ge=1)
    chunk_size: int = Field(512, alias="RAG_CHUNK_SIZE", ge=64)
    chunk_overlap: int = Field(64, alias="RAG_CHUNK_OVERLAP", ge=0)
    similarity_threshold: float = Field(
        0.35, alias="RAG_SIMILARITY_THRESHOLD", ge=0.0, le=1.0
    )

    @field_validator("rerank_top_k")
    @classmethod
    def rerank_leq_top_k(cls, v: int, info) -> int:
        top_k = info.data.get("top_k", 10)
        if v > top_k:
            raise ValueError(
                f"rerank_top_k ({v}) must be ≤ top_k ({top_k})"
            )
        return v

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


class IngestionSettings(BaseSettings):
    """Document ingestion pipeline configuration."""

    data_dir: Path = Field(
        PROJECT_ROOT / "data" / "raw", alias="INGESTION_DATA_DIR"
    )
    processed_dir: Path = Field(
        PROJECT_ROOT / "data" / "processed", alias="INGESTION_PROCESSED_DIR"
    )
    batch_size: int = Field(50, alias="INGESTION_BATCH_SIZE", ge=1)

    @field_validator("data_dir", "processed_dir", mode="before")
    @classmethod
    def resolve_and_create(cls, v: str | Path) -> Path:
        p = Path(v)
        p.mkdir(parents=True, exist_ok=True)
        return p

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        "INFO", alias="LOG_LEVEL"
    )
    format: Literal["json", "console"] = Field("json", alias="LOG_FORMAT")
    file: Path = Field(PROJECT_ROOT / "logs" / "app.log", alias="LOG_FILE")
    rotation: str = Field("10 MB", alias="LOG_ROTATION")
    retention: str = Field("30 days", alias="LOG_RETENTION")

    @field_validator("file", mode="before")
    @classmethod
    def ensure_log_dir(cls, v: str | Path) -> Path:
        p = Path(v)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


class SecuritySettings(BaseSettings):
    """Auth and rate limiting configuration."""

    jwt_algorithm: str = Field("HS256", alias="JWT_ALGORITHM")
    jwt_expire_minutes: int = Field(60, alias="JWT_EXPIRE_MINUTES", ge=1)
    rate_limit_per_minute: int = Field(30, alias="RATE_LIMIT_PER_MINUTE", ge=1)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


# ── Aggregate settings object ─────────────────────────────────────────────────

class Settings:
    """
    Aggregated settings namespace.
    Access via: settings.llm.temperature, settings.chroma.collection_name, etc.
    """

    def __init__(self) -> None:
        self.app = AppSettings()
        self.server = ServerSettings()
        self.llm = LLMSettings()
        self.embedding = EmbeddingSettings()
        self.chroma = ChromaSettings()
        self.rag = RAGSettings()
        self.ingestion = IngestionSettings()
        self.logging = LoggingSettings()
        self.security = SecuritySettings()

    def is_production(self) -> bool:
        return self.app.env == "production"

    def is_debug(self) -> bool:
        return self.app.debug

    def __repr__(self) -> str:
        return (
            f"Settings(env={self.app.env!r}, "
            f"model={self.llm.ollama_model!r}, "
            f"collection={self.chroma.collection_name!r})"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns a cached singleton Settings instance.
    Call invalidate_settings_cache() in tests to reset.
    """
    return Settings()


def invalidate_settings_cache() -> None:
    """Clear the settings cache — useful in tests that override env vars."""
    get_settings.cache_clear()
