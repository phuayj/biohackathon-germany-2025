"""Configuration management for the unified data loader.

This module handles loading configuration from environment variables
and .env files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Loader configuration loaded from environment."""

    # Neo4j connection
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str

    # API keys (optional)
    ncbi_api_key: str | None = None
    disgenet_api_key: str | None = None
    cosmic_email: str | None = None
    cosmic_password: str | None = None

    # Paths
    data_dir: Path = field(default_factory=lambda: Path("./data"))

    # Options
    cosmic_version: str = "v103"
    reactome_species: str = "Homo sapiens"
    batch_size: int = 5000

    # Retry settings
    max_retries: int = 3
    retry_initial_delay: float = 2.0
    retry_max_delay: float = 60.0

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> Config:
        """Load configuration from environment variables and optional .env file.

        Args:
            env_file: Optional path to .env file. If None, looks for .env in cwd.

        Returns:
            Config instance with values from environment.

        Raises:
            ValueError: If required configuration is missing.
        """
        # Load .env file if present
        if env_file is None:
            env_file = Path(".env")

        if env_file.exists():
            _load_dotenv(env_file)

        # Required values
        neo4j_uri = os.environ.get("NEO4J_URI")
        neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
        neo4j_password = os.environ.get("NEO4J_PASSWORD")

        if not neo4j_uri:
            raise ValueError("NEO4J_URI environment variable is required")
        if not neo4j_password:
            raise ValueError("NEO4J_PASSWORD environment variable is required")

        # Optional values with defaults
        data_dir = Path(os.environ.get("DATA_DIR", "./data"))

        return cls(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            ncbi_api_key=os.environ.get("NCBI_API_KEY") or None,
            disgenet_api_key=os.environ.get("DISGENET_API_KEY") or None,
            cosmic_email=os.environ.get("COSMIC_EMAIL") or None,
            cosmic_password=os.environ.get("COSMIC_PASSWORD") or None,
            data_dir=data_dir,
            cosmic_version=os.environ.get("COSMIC_VERSION", "v103"),
            reactome_species=os.environ.get("REACTOME_SPECIES", "Homo sapiens"),
            batch_size=int(os.environ.get("NEO4J_BATCH_SIZE", "5000")),
            max_retries=int(os.environ.get("MAX_RETRIES", "3")),
            retry_initial_delay=float(os.environ.get("RETRY_INITIAL_DELAY", "2.0")),
            retry_max_delay=float(os.environ.get("RETRY_MAX_DELAY", "60.0")),
        )

    def has_cosmic_credentials(self) -> bool:
        """Check if COSMIC credentials are configured."""
        return bool(self.cosmic_email and self.cosmic_password)

    def has_ncbi_api_key(self) -> bool:
        """Check if NCBI API key is configured."""
        return bool(self.ncbi_api_key)

    def has_disgenet_api_key(self) -> bool:
        """Check if DisGeNET API key is configured."""
        return bool(self.disgenet_api_key)


def _load_dotenv(path: Path) -> None:
    """Simple .env file loader (no external dependencies).

    Parses KEY=VALUE lines, ignoring comments and empty lines.
    Does not override existing environment variables.
    """
    try:
        with path.open() as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue
                # Parse KEY=VALUE
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    # Don't override existing env vars
                    if key and key not in os.environ:
                        os.environ[key] = value
    except OSError:
        pass  # Ignore errors reading .env file
