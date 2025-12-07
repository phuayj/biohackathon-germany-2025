"""Tests for loader registry module."""

from __future__ import annotations

from typing import Literal

import pytest

from nerve.loader.config import Config
from nerve.loader.protocol import LoadStats
from nerve.loader.registry import SourceRegistry


def make_test_config() -> Config:
    """Create a test config."""
    return Config(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
    )


class MockSource:
    """Mock data source for testing."""

    def __init__(
        self,
        name: str,
        stage: int,
        dependencies: list[str] | None = None,
        requires_credentials: list[str] | None = None,
    ) -> None:
        self.name = name
        self.display_name = name.title()
        self.stage = stage
        self.dependencies = dependencies or []
        self.requires_credentials = requires_credentials or []

    def check_credentials(self, config: Config) -> tuple[bool, str | None]:
        return True, None

    def download(self, config: Config, force: bool = False) -> None:
        pass

    def load(
        self,
        driver: object,
        config: Config,
        mode: Literal["replace", "merge"],
    ) -> LoadStats:
        return LoadStats(source=self.name)


class TestSourceRegistry:
    """Tests for SourceRegistry class."""

    def test_register_source(self) -> None:
        """Test registering a source."""
        registry = SourceRegistry()
        source = MockSource("test", stage=1)

        registry.register(source)

        assert registry.get("test") is source

    def test_register_duplicate_raises_error(self) -> None:
        """Test that registering duplicate source raises error."""
        registry = SourceRegistry()
        source1 = MockSource("test", stage=1)
        source2 = MockSource("test", stage=2)

        registry.register(source1)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(source2)

    def test_get_nonexistent_returns_none(self) -> None:
        """Test getting nonexistent source returns None."""
        registry = SourceRegistry()

        assert registry.get("nonexistent") is None

    def test_all_sources(self) -> None:
        """Test getting all registered sources."""
        registry = SourceRegistry()
        source1 = MockSource("source1", stage=1)
        source2 = MockSource("source2", stage=2)

        registry.register(source1)
        registry.register(source2)

        sources = registry.all_sources()
        assert len(sources) == 2
        assert source1 in sources
        assert source2 in sources

    def test_get_sources_by_stage(self) -> None:
        """Test getting sources by stage."""
        registry = SourceRegistry()
        source1 = MockSource("source1", stage=1)
        source2 = MockSource("source2", stage=1)
        source3 = MockSource("source3", stage=2)

        registry.register(source1)
        registry.register(source2)
        registry.register(source3)

        stage1 = registry.get_sources_by_stage(1)
        stage2 = registry.get_sources_by_stage(2)
        stage3 = registry.get_sources_by_stage(3)

        assert len(stage1) == 2
        assert len(stage2) == 1
        assert len(stage3) == 0

    def test_get_source_names(self) -> None:
        """Test getting all source names."""
        registry = SourceRegistry()
        source1 = MockSource("alpha", stage=1)
        source2 = MockSource("beta", stage=2)

        registry.register(source1)
        registry.register(source2)

        names = registry.get_source_names()
        assert "alpha" in names
        assert "beta" in names


class TestResolveExecutionOrder:
    """Tests for resolve_execution_order method."""

    def test_resolve_basic_stages(self) -> None:
        """Test basic stage resolution."""
        registry = SourceRegistry()
        registry.register(MockSource("s1", stage=1))
        registry.register(MockSource("s2", stage=2))
        registry.register(MockSource("s3", stage=3))

        stages = registry.resolve_execution_order()

        assert len(stages) == 3
        assert stages[0][0].name == "s1"
        assert stages[1][0].name == "s2"
        assert stages[2][0].name == "s3"

    def test_resolve_multiple_sources_per_stage(self) -> None:
        """Test resolution with multiple sources per stage."""
        registry = SourceRegistry()
        registry.register(MockSource("a", stage=1))
        registry.register(MockSource("b", stage=1))
        registry.register(MockSource("c", stage=2))

        stages = registry.resolve_execution_order()

        assert len(stages) == 2
        assert len(stages[0]) == 2  # Two sources in stage 1
        assert len(stages[1]) == 1  # One source in stage 2

    def test_resolve_with_selected_sources(self) -> None:
        """Test resolution with selected sources filter."""
        registry = SourceRegistry()
        registry.register(MockSource("a", stage=1))
        registry.register(MockSource("b", stage=1))
        registry.register(MockSource("c", stage=2))

        stages = registry.resolve_execution_order(selected_sources=["a", "c"])

        # Should only include a and c
        all_names = [s.name for stage in stages for s in stage]
        assert "a" in all_names
        assert "c" in all_names
        assert "b" not in all_names

    def test_resolve_with_skip_sources(self) -> None:
        """Test resolution with skipped sources."""
        registry = SourceRegistry()
        registry.register(MockSource("a", stage=1))
        registry.register(MockSource("b", stage=1))
        registry.register(MockSource("c", stage=2))

        stages = registry.resolve_execution_order(skip_sources=["b"])

        all_names = [s.name for stage in stages for s in stage]
        assert "a" in all_names
        assert "c" in all_names
        assert "b" not in all_names

    def test_resolve_with_stages_filter(self) -> None:
        """Test resolution with stages filter."""
        registry = SourceRegistry()
        registry.register(MockSource("a", stage=1))
        registry.register(MockSource("b", stage=2))
        registry.register(MockSource("c", stage=3))

        stages = registry.resolve_execution_order(stages=[1, 3])

        all_names = [s.name for stage in stages for s in stage]
        assert "a" in all_names
        assert "c" in all_names
        assert "b" not in all_names

    def test_resolve_empty_stages_removed(self) -> None:
        """Test that empty stages are removed from result."""
        registry = SourceRegistry()
        registry.register(MockSource("a", stage=1))
        registry.register(MockSource("c", stage=3))
        # Stage 2 has no sources

        stages = registry.resolve_execution_order()

        # Should only have 2 stages (1 and 3), not 3
        assert len(stages) == 2

    def test_resolve_with_dependencies_sorted(self) -> None:
        """Test that sources with dependencies are sorted correctly within stage."""
        registry = SourceRegistry()
        # b depends on a, both in stage 1
        registry.register(MockSource("b", stage=1, dependencies=["a"]))
        registry.register(MockSource("a", stage=1))

        stages = registry.resolve_execution_order()

        # a should come before b within stage 1
        stage1_names = [s.name for s in stages[0]]
        assert stage1_names.index("a") < stage1_names.index("b")


class TestCheckAllCredentials:
    """Tests for check_all_credentials method."""

    def test_all_credentials_available(self) -> None:
        """Test when all credentials are available."""
        registry = SourceRegistry()
        registry.register(MockSource("s1", stage=1))

        config = make_test_config()
        results = registry.check_all_credentials(config)

        assert len(results) == 1
        assert results["s1"] == (True, None)

    def test_missing_credentials_detected(self) -> None:
        """Test detection of missing credentials."""

        class SourceWithCreds:
            name = "cosmic"
            display_name = "COSMIC"
            stage = 2
            dependencies: list[str] = []
            requires_credentials = ["COSMIC_EMAIL", "COSMIC_PASSWORD"]

            def check_credentials(self, config: Config) -> tuple[bool, str | None]:
                if not config.cosmic_email or not config.cosmic_password:
                    return False, "Missing COSMIC credentials"
                return True, None

            def download(self, config: Config, force: bool = False) -> None:
                pass

            def load(
                self, driver: object, config: Config, mode: Literal["replace", "merge"]
            ) -> LoadStats:
                return LoadStats(source=self.name)

        registry = SourceRegistry()
        registry.register(SourceWithCreds())

        config = make_test_config()  # No COSMIC credentials
        results = registry.check_all_credentials(config)

        assert len(results) == 1
        available, reason = results["cosmic"]
        assert available is False
        assert reason is not None
        assert "Missing COSMIC credentials" in reason
