"""NERVE Unified Data Loader.

A single entry point for loading all biomedical data into Neo4j.

Usage:
    uv run python -m nerve.loader
    uv run python -m nerve.loader --sources monarch,hpo
    uv run python -m nerve.loader --merge
    uv run python -m nerve.loader --dry-run
"""

from nerve.loader.config import Config
from nerve.loader.protocol import DataSource, LoadStats

__all__ = ["Config", "DataSource", "LoadStats"]
