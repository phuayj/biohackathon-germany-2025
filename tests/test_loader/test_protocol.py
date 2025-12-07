"""Tests for loader protocol module."""

from __future__ import annotations

from nerve.loader.protocol import LoadStats


class TestLoadStats:
    """Tests for LoadStats dataclass."""

    def test_loadstats_creation_defaults(self) -> None:
        """Test LoadStats with default values."""
        stats = LoadStats(source="test_source")
        assert stats.source == "test_source"
        assert stats.nodes_created == 0
        assert stats.edges_created == 0
        assert stats.nodes_updated == 0
        assert stats.edges_updated == 0
        assert stats.duration_seconds == 0.0
        assert stats.skipped is False
        assert stats.skip_reason is None
        assert stats.extra == {}

    def test_loadstats_creation_with_values(self) -> None:
        """Test LoadStats with all values."""
        stats = LoadStats(
            source="monarch",
            nodes_created=1000,
            edges_created=5000,
            nodes_updated=100,
            edges_updated=500,
            duration_seconds=45.5,
            extra={"publications": 200},
        )
        assert stats.source == "monarch"
        assert stats.nodes_created == 1000
        assert stats.edges_created == 5000
        assert stats.nodes_updated == 100
        assert stats.edges_updated == 500
        assert stats.duration_seconds == 45.5
        assert stats.extra == {"publications": 200}

    def test_loadstats_skipped(self) -> None:
        """Test LoadStats when skipped."""
        stats = LoadStats(
            source="cosmic",
            skipped=True,
            skip_reason="Missing credentials",
        )
        assert stats.skipped is True
        assert stats.skip_reason == "Missing credentials"

    def test_loadstats_str_normal(self) -> None:
        """Test __str__ for normal load."""
        stats = LoadStats(
            source="monarch",
            nodes_created=1000,
            edges_created=5000,
            duration_seconds=45.5,
        )
        result = str(stats)
        assert "monarch" in result
        assert "1,000 nodes" in result
        assert "5,000 edges" in result
        assert "45.5s" in result

    def test_loadstats_str_skipped(self) -> None:
        """Test __str__ for skipped load."""
        stats = LoadStats(
            source="cosmic",
            skipped=True,
            skip_reason="Missing credentials",
        )
        result = str(stats)
        assert "cosmic" in result
        assert "skipped" in result
        assert "Missing credentials" in result

    def test_loadstats_str_no_changes(self) -> None:
        """Test __str__ when no changes made."""
        stats = LoadStats(source="empty", duration_seconds=1.0)
        result = str(stats)
        assert "empty" in result
        assert "no changes" in result

    def test_loadstats_str_with_updates(self) -> None:
        """Test __str__ when updates made."""
        stats = LoadStats(
            source="update_test",
            nodes_updated=50,
            edges_updated=100,
            duration_seconds=10.0,
        )
        result = str(stats)
        assert "update_test" in result
        assert "50 nodes updated" in result
        assert "100 edges updated" in result
