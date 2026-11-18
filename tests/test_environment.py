"""Task 1 smoke tests: the environment and the configuration layer work."""

from __future__ import annotations

import importlib
import sys

import pytest

REQUIRED_MODULES = [
    "celery",
    "redis",
    "requests",
    "bs4",
    "lxml",
    "trafilatura",
    "tenacity",
    "chromadb",
    "fastembed",
    "numpy",
    "anthropic",
    "fastapi",
    "uvicorn",
    "pydantic",
    "prometheus_client",
]


def test_python_version_is_supported() -> None:
    assert sys.version_info >= (3, 11), "Python 3.11+ is required"


@pytest.mark.parametrize("module", REQUIRED_MODULES)
def test_required_dependency_is_importable(module: str) -> None:
    assert importlib.import_module(module) is not None


def test_settings_load_with_defaults() -> None:
    from src.common.config import settings

    assert settings.llm_model
    assert settings.celery_broker_url.startswith("redis://")
    assert settings.chroma_port > 0
    assert settings.scraper_max_retries >= 1


def test_logging_setup_tags_component() -> None:
    from src.common.logging_config import setup_logging

    logger = setup_logging("unit-test")
    assert logger.name == "unit-test"
    assert logging_handlers_configured()


def logging_handlers_configured() -> bool:
    import logging

    return bool(logging.getLogger().handlers)
