"""Tests for loader protocol module."""

from __future__ import annotations

import io
import sys

from nerve.loader.protocol import LoadStats, ProgressReporter


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


class TestProgressReporter:
    """Tests for ProgressReporter class."""

    def test_progress_reporter_creation(self) -> None:
        """Test ProgressReporter initialization."""
        reporter = ProgressReporter(
            source_name="test",
            operation="Loading",
            report_interval=1000,
            verbose=True,
        )
        assert reporter.source_name == "test"
        assert reporter.operation == "Loading"
        assert reporter.report_interval == 1000
        assert reporter.verbose is True

    def test_progress_reporter_no_output_when_not_verbose(self) -> None:
        """Test that no output is produced when verbose=False."""
        reporter = ProgressReporter(
            source_name="test",
            operation="Loading",
            report_interval=10,
            verbose=False,
        )

        # Capture stdout
        captured = io.StringIO()
        sys.stdout = captured
        try:
            for i in range(100):
                reporter.update()
            reporter.finish()
        finally:
            sys.stdout = sys.__stdout__

        assert captured.getvalue() == ""

    def test_progress_reporter_reports_at_interval(self) -> None:
        """Test that progress is reported at intervals."""
        reporter = ProgressReporter(
            source_name="test",
            operation="Processing",
            report_interval=10,
            verbose=True,
        )

        captured = io.StringIO()
        sys.stdout = captured
        try:
            for i in range(25):
                reporter.update()
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        # Should have reported at 10 and 20
        assert output.count("[test]") == 2
        assert "10" in output
        assert "20" in output

    def test_progress_reporter_finish_reports_final_count(self) -> None:
        """Test that finish reports the final count."""
        reporter = ProgressReporter(
            source_name="test",
            operation="Loading",
            report_interval=100,
            verbose=True,
        )

        captured = io.StringIO()
        sys.stdout = captured
        try:
            for i in range(50):
                reporter.update()
            reporter.finish()
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        assert "50" in output
        assert "complete" in output

    def test_progress_reporter_update_with_absolute_count(self) -> None:
        """Test update with absolute count."""
        reporter = ProgressReporter(
            source_name="test",
            operation="Loading",
            report_interval=10,
            verbose=True,
        )

        captured = io.StringIO()
        sys.stdout = captured
        try:
            reporter.update(count=15)
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        assert "15" in output
