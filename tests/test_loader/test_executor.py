"""Tests for loader executor module."""

from __future__ import annotations

from typing import Literal
from unittest.mock import Mock


from nerve.loader.config import Config
from nerve.loader.executor import StageExecutor
from nerve.loader.protocol import LoadStats


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
        nodes: int = 100,
        edges: int = 200,
        should_fail: bool = False,
        fail_on_download: bool = False,
        missing_creds: bool = False,
    ) -> None:
        self.name = name
        self.display_name = name.title()
        self.stage = stage
        self.dependencies: list[str] = []
        self.requires_credentials: list[str] = []
        self._nodes = nodes
        self._edges = edges
        self._should_fail = should_fail
        self._fail_on_download = fail_on_download
        self._missing_creds = missing_creds
        self.download_called = False
        self.load_called = False

    def check_credentials(self, config: Config) -> tuple[bool, str | None]:
        if self._missing_creds:
            return False, "Missing required credentials"
        return True, None

    def download(self, config: Config, force: bool = False) -> None:
        self.download_called = True
        if self._fail_on_download:
            raise RuntimeError("Download failed")

    def load(
        self,
        driver: object,
        config: Config,
        mode: Literal["replace", "merge"],
    ) -> LoadStats:
        self.load_called = True
        if self._should_fail:
            raise RuntimeError("Load failed")
        return LoadStats(
            source=self.name,
            nodes_created=self._nodes,
            edges_created=self._edges,
        )


class TestStageExecutor:
    """Tests for StageExecutor class."""

    def test_executor_creation(self) -> None:
        """Test executor initialization."""
        config = make_test_config()
        driver = Mock()

        executor = StageExecutor(config, driver, mode="replace", verbose=False)

        assert executor.config is config
        assert executor.driver is driver
        assert executor.mode == "replace"
        assert executor.parallel is True

    def test_execute_single_stage(self) -> None:
        """Test executing a single stage."""
        config = make_test_config()
        driver = Mock()
        source = MockSource("test", stage=1)

        executor = StageExecutor(config, driver, mode="replace", verbose=False)
        stats = executor.execute_stages([[source]])

        assert len(stats) == 1
        assert stats[0].source == "test"
        assert stats[0].nodes_created == 100
        assert stats[0].edges_created == 200
        assert source.download_called is True
        assert source.load_called is True

    def test_execute_multiple_stages(self) -> None:
        """Test executing multiple stages in order."""
        config = make_test_config()
        driver = Mock()
        source1 = MockSource("s1", stage=1, nodes=10, edges=20)
        source2 = MockSource("s2", stage=2, nodes=30, edges=40)

        executor = StageExecutor(config, driver, mode="replace", verbose=False)
        stats = executor.execute_stages([[source1], [source2]])

        assert len(stats) == 2
        assert stats[0].source == "s1"
        assert stats[1].source == "s2"

    def test_execute_download_only(self) -> None:
        """Test download-only mode."""
        config = make_test_config()
        driver = Mock()
        source = MockSource("test", stage=1)

        executor = StageExecutor(config, driver, mode="replace", verbose=False)
        stats = executor.execute_stages([[source]], download_only=True)

        assert len(stats) == 1
        assert source.download_called is True
        assert source.load_called is False

    def test_execute_skip_download(self) -> None:
        """Test skipping download phase."""
        config = make_test_config()
        driver = Mock()
        source = MockSource("test", stage=1)

        executor = StageExecutor(config, driver, mode="replace", verbose=False)
        stats = executor.execute_stages([[source]], skip_download=True)

        assert len(stats) == 1
        assert source.download_called is False
        assert source.load_called is True

    def test_execute_with_sample(self) -> None:
        """Test sample parameter is passed to config."""
        config = make_test_config()
        driver = Mock()

        class SampleCheckSource(MockSource):
            def load(
                self,
                driver: object,
                config: Config,
                mode: Literal["replace", "merge"],
            ) -> LoadStats:
                # Check that sample is set in config
                sample = getattr(config, "_sample", None)
                return LoadStats(
                    source=self.name,
                    nodes_created=sample if sample else 100,
                )

        source = SampleCheckSource("test", stage=1)

        executor = StageExecutor(config, driver, mode="replace", verbose=False)
        stats = executor.execute_stages([[source]], sample=50)

        assert stats[0].nodes_created == 50

    def test_execute_missing_credentials_skips(self) -> None:
        """Test that missing credentials cause source to be skipped."""
        config = make_test_config()
        driver = Mock()
        source = MockSource("test", stage=1, missing_creds=True)

        executor = StageExecutor(config, driver, mode="replace", verbose=False)
        stats = executor.execute_stages([[source]])

        assert len(stats) == 1
        assert stats[0].skipped is True
        assert "Missing required credentials" in (stats[0].skip_reason or "")
        assert source.download_called is False
        assert source.load_called is False

    def test_execute_download_failure(self) -> None:
        """Test handling of download failure."""
        config = make_test_config()
        driver = Mock()
        source = MockSource("test", stage=1, fail_on_download=True)

        executor = StageExecutor(config, driver, mode="replace", verbose=False)
        stats = executor.execute_stages([[source]])

        assert len(stats) == 1
        assert stats[0].skipped is True
        assert "Download failed" in (stats[0].skip_reason or "")

    def test_execute_load_failure(self) -> None:
        """Test handling of load failure."""
        config = make_test_config()
        driver = Mock()
        source = MockSource("test", stage=1, should_fail=True)

        executor = StageExecutor(config, driver, mode="replace", verbose=False)
        stats = executor.execute_stages([[source]])

        assert len(stats) == 1
        assert stats[0].skipped is True
        assert "Load failed" in (stats[0].skip_reason or "")


class TestParallelExecution:
    """Tests for parallel execution within stages."""

    def test_parallel_execution_multiple_sources(self) -> None:
        """Test that multiple sources in a stage run in parallel."""
        config = make_test_config()
        driver = Mock()
        source1 = MockSource("s1", stage=1, nodes=10)
        source2 = MockSource("s2", stage=1, nodes=20)
        source3 = MockSource("s3", stage=1, nodes=30)

        executor = StageExecutor(config, driver, mode="replace", parallel=True, verbose=False)
        stats = executor.execute_stages([[source1, source2, source3]])

        assert len(stats) == 3
        # All sources should have been executed
        names = {s.source for s in stats}
        assert names == {"s1", "s2", "s3"}

    def test_sequential_execution_when_disabled(self) -> None:
        """Test sequential execution when parallel is disabled."""
        config = make_test_config()
        driver = Mock()
        source1 = MockSource("s1", stage=1)
        source2 = MockSource("s2", stage=1)

        executor = StageExecutor(config, driver, mode="replace", parallel=False, verbose=False)
        stats = executor.execute_stages([[source1, source2]])

        assert len(stats) == 2
        # Both sources executed
        assert source1.load_called is True
        assert source2.load_called is True

    def test_parallel_handles_partial_failure(self) -> None:
        """Test that parallel execution handles partial failures gracefully."""
        config = make_test_config()
        driver = Mock()
        source1 = MockSource("s1", stage=1, nodes=10)
        source2 = MockSource("s2", stage=1, should_fail=True)
        source3 = MockSource("s3", stage=1, nodes=30)

        executor = StageExecutor(config, driver, mode="replace", parallel=True, verbose=False)
        stats = executor.execute_stages([[source1, source2, source3]])

        assert len(stats) == 3
        # s1 and s3 should succeed
        successful = [s for s in stats if not s.skipped]
        failed = [s for s in stats if s.skipped]
        assert len(successful) == 2
        assert len(failed) == 1
        assert failed[0].source == "s2"


class TestMergeMode:
    """Tests for merge vs replace mode."""

    def test_replace_mode_passed_to_source(self) -> None:
        """Test that replace mode is passed to source.load()."""
        config = make_test_config()
        driver = Mock()

        class ModeCheckSource(MockSource):
            received_mode: str | None = None

            def load(
                self,
                driver: object,
                config: Config,
                mode: Literal["replace", "merge"],
            ) -> LoadStats:
                self.received_mode = mode
                return LoadStats(source=self.name)

        source = ModeCheckSource("test", stage=1)

        executor = StageExecutor(config, driver, mode="replace", verbose=False)
        executor.execute_stages([[source]])

        assert source.received_mode == "replace"

    def test_merge_mode_passed_to_source(self) -> None:
        """Test that merge mode is passed to source.load()."""
        config = make_test_config()
        driver = Mock()

        class ModeCheckSource(MockSource):
            received_mode: str | None = None

            def load(
                self,
                driver: object,
                config: Config,
                mode: Literal["replace", "merge"],
            ) -> LoadStats:
                self.received_mode = mode
                return LoadStats(source=self.name)

        source = ModeCheckSource("test", stage=1)

        executor = StageExecutor(config, driver, mode="merge", verbose=False)
        executor.execute_stages([[source]])

        assert source.received_mode == "merge"
