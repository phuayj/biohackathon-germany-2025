"""Tests for the pipeline orchestrator."""

import pytest
from kg_skeptic.pipeline import SkepticPipeline


class TestSkepticPipeline:
    def test_init_defaults(self) -> None:
        """Test initialization with default arguments."""
        pipeline = SkepticPipeline()
        assert pipeline.config == {}

    def test_init_with_config(self) -> None:
        """Test initialization with a configuration dictionary."""
        config = {"verbose": True, "model": "gpt-4"}
        pipeline = SkepticPipeline(config=config)
        assert pipeline.config == config

    def test_run_raises_not_implemented(self) -> None:
        """Test that run() raises NotImplementedError (for now)."""
        pipeline = SkepticPipeline()
        with pytest.raises(NotImplementedError):
            pipeline.run({})
