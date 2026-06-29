"""
utils/env_check.py
───────────────────
Pre-flight checks run at application startup.
Validates that all required services are reachable and config is sane.

Usage:
    from utils.env_check import run_preflight_checks
    await run_preflight_checks()   # raises EnvironmentError on failure
"""

import asyncio
import sys
from pathlib import Path

import httpx

from config.settings import get_settings
from app_logging.logger import get_logger

log = get_logger(__name__)


class PreflightError(RuntimeError):
    """Raised when a critical preflight check fails."""


async def check_ollama(base_url: str, model: str, timeout: int = 10) -> None:
    """Verify Ollama is running and the target model is available."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(f"{base_url}/api/tags")
            r.raise_for_status()
            available = [m["name"] for m in r.json().get("models", [])]
            # Accept both "llama3.1:8b" and "llama3.1:8b-instruct-q4_0" etc.
            model_prefix = model.split(":")[0]
            found = any(m.startswith(model_prefix) for m in available)
            if not found:
                log.warning(
                    "ollama.model_missing",
                    model=model,
                    available=available,
                    hint=f"Run: ollama pull {model}",
                )
            else:
                log.info("ollama.ok", model=model, url=base_url)
    except httpx.ConnectError:
        raise PreflightError(
            f"Cannot reach Ollama at {base_url}. "
            "Is it running? Start with: ollama serve"
        )
    except Exception as exc:
        raise PreflightError(f"Ollama check failed: {exc}") from exc


def check_chroma_dir(persist_dir: Path) -> None:
    """Ensure the ChromaDB persistence directory exists and is writable."""
    persist_dir.mkdir(parents=True, exist_ok=True)
    test_file = persist_dir / ".write_test"
    try:
        test_file.touch()
        test_file.unlink()
        log.info("chroma.dir_ok", path=str(persist_dir))
    except OSError as exc:
        raise PreflightError(
            f"ChromaDB directory {persist_dir} is not writable: {exc}"
        ) from exc


def check_python_version(min_major: int = 3, min_minor: int = 11) -> None:
    """Enforce minimum Python version."""
    major, minor = sys.version_info.major, sys.version_info.minor
    if (major, minor) < (min_major, min_minor):
        raise PreflightError(
            f"Python {min_major}.{min_minor}+ is required, got {major}.{minor}."
        )
    log.info("python.ok", version=f"{major}.{minor}")


def check_env_file() -> None:
    """.env must exist (not just .env.example)."""
    if not Path(".env").exists():
        raise PreflightError(
            ".env file not found. Copy .env.example to .env and fill in values."
        )
    log.info("env_file.ok")


async def run_preflight_checks(*, skip_ollama: bool = False) -> None:
    """
    Run all preflight checks. Call once at application startup.

    Args:
        skip_ollama: Set True in tests to skip the Ollama connectivity check.
    """
    settings = get_settings()
    log.info("preflight.start", env=settings.app.env)

    check_python_version()
    check_env_file()
    check_chroma_dir(settings.chroma.persist_dir)

    if not skip_ollama:
        await check_ollama(
            base_url=settings.llm.ollama_base_url,
            model=settings.llm.ollama_model,
            timeout=settings.llm.ollama_timeout,
        )

    log.info("preflight.complete")


if __name__ == "__main__":
    asyncio.run(run_preflight_checks())
