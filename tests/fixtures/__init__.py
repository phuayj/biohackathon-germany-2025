"""Test fixtures for NERVE."""

from pathlib import Path

FIXTURES_DIR = Path(__file__).parent


def load_fixture(name: str) -> str:
    """Load a fixture file by name."""
    return (FIXTURES_DIR / name).read_text()
