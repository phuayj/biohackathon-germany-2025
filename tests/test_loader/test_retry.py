"""Tests for loader retry module."""

from __future__ import annotations

from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

import pytest

from nerve.loader.retry import (
    DownloadError,
    FatalLoadError,
    LoadError,
    Neo4jTransientError,
    RateLimitError,
    RetryHandler,
    is_neo4j_transient_error,
    with_retry,
)


class TestRetryHandler:
    """Tests for RetryHandler class."""

    def test_execute_success_first_try(self) -> None:
        """Test successful execution on first try."""
        handler = RetryHandler(max_retries=3, verbose=False)
        mock_func = Mock(return_value="success")

        result = handler.execute(mock_func, "test_op")

        assert result == "success"
        assert mock_func.call_count == 1

    def test_execute_success_after_retry(self) -> None:
        """Test successful execution after retries."""
        handler = RetryHandler(max_retries=3, initial_delay=0.01, verbose=False)
        mock_func = Mock(side_effect=[URLError("fail"), URLError("fail"), "success"])

        result = handler.execute(mock_func, "test_op")

        assert result == "success"
        assert mock_func.call_count == 3

    def test_execute_fails_after_max_retries(self) -> None:
        """Test failure after exhausting all retries."""
        handler = RetryHandler(max_retries=2, initial_delay=0.01, verbose=False)
        mock_func = Mock(side_effect=URLError("persistent failure"))

        with pytest.raises(FatalLoadError, match="persistent failure"):
            handler.execute(mock_func, "test_op")

        # Initial attempt + 2 retries = 3 calls
        assert mock_func.call_count == 3

    def test_execute_http_4xx_no_retry(self) -> None:
        """Test that HTTP 4xx errors (except 429) don't trigger retry."""
        handler = RetryHandler(max_retries=3, initial_delay=0.01, verbose=False)
        mock_func = Mock(
            side_effect=HTTPError(url="http://test", code=404, msg="Not Found", hdrs=None, fp=None)  # type: ignore[arg-type]
        )

        with pytest.raises(FatalLoadError, match="HTTP 404"):
            handler.execute(mock_func, "test_op")

        # Should fail immediately, no retries
        assert mock_func.call_count == 1

    def test_execute_http_429_retries(self) -> None:
        """Test that HTTP 429 (rate limit) triggers retry."""
        handler = RetryHandler(max_retries=2, initial_delay=0.01, verbose=False)
        mock_func = Mock(
            side_effect=[
                HTTPError(
                    url="http://test",
                    code=429,
                    msg="Too Many Requests",
                    hdrs=None,  # type: ignore[arg-type]
                    fp=None,
                ),
                "success",
            ]
        )

        result = handler.execute(mock_func, "test_op")

        assert result == "success"
        assert mock_func.call_count == 2

    def test_execute_http_5xx_retries(self) -> None:
        """Test that HTTP 5xx errors trigger retry."""
        handler = RetryHandler(max_retries=2, initial_delay=0.01, verbose=False)
        mock_func = Mock(
            side_effect=[
                HTTPError(
                    url="http://test",
                    code=500,
                    msg="Server Error",
                    hdrs=None,  # type: ignore[arg-type]
                    fp=None,
                ),
                "success",
            ]
        )

        result = handler.execute(mock_func, "test_op")

        assert result == "success"
        assert mock_func.call_count == 2

    def test_execute_exponential_backoff(self) -> None:
        """Test that delays follow exponential backoff."""
        handler = RetryHandler(max_retries=2, initial_delay=0.1, max_delay=10.0, verbose=False)
        mock_func = Mock(side_effect=[URLError("fail"), URLError("fail"), "success"])

        with patch("time.sleep") as mock_sleep:
            handler.execute(mock_func, "test_op")

        # First retry: 0.1s, second retry: 0.2s
        assert mock_sleep.call_count == 2
        calls = [call[0][0] for call in mock_sleep.call_args_list]
        assert calls[0] == pytest.approx(0.1, rel=0.1)
        assert calls[1] == pytest.approx(0.2, rel=0.1)

    def test_execute_max_delay_cap(self) -> None:
        """Test that delay is capped at max_delay."""
        handler = RetryHandler(max_retries=5, initial_delay=10.0, max_delay=15.0, verbose=False)
        mock_func = Mock(side_effect=[URLError("fail")] * 5 + ["success"])

        with patch("time.sleep") as mock_sleep:
            handler.execute(mock_func, "test_op")

        # Check that delays are capped at 15.0
        calls = [call[0][0] for call in mock_sleep.call_args_list]
        for delay in calls:
            assert delay <= 15.0

    def test_execute_rate_limit_error_retries(self) -> None:
        """Test that RateLimitError triggers retry."""
        handler = RetryHandler(max_retries=2, initial_delay=0.01, verbose=False)
        mock_func = Mock(side_effect=[RateLimitError("rate limited"), "success"])

        result = handler.execute(mock_func, "test_op")

        assert result == "success"
        assert mock_func.call_count == 2

    def test_execute_connection_error_retries(self) -> None:
        """Test that ConnectionError triggers retry."""
        handler = RetryHandler(max_retries=2, initial_delay=0.01, verbose=False)
        mock_func = Mock(side_effect=[ConnectionError("connection lost"), "success"])

        result = handler.execute(mock_func, "test_op")

        assert result == "success"

    def test_execute_timeout_error_retries(self) -> None:
        """Test that TimeoutError triggers retry."""
        handler = RetryHandler(max_retries=2, initial_delay=0.01, verbose=False)
        mock_func = Mock(side_effect=[TimeoutError("timed out"), "success"])

        result = handler.execute(mock_func, "test_op")

        assert result == "success"


class TestWithRetry:
    """Tests for with_retry convenience function."""

    def test_with_retry_success(self) -> None:
        """Test with_retry success."""
        mock_func = Mock(return_value="success")

        result = with_retry(mock_func, "test_op", verbose=False)

        assert result == "success"

    def test_with_retry_with_retries(self) -> None:
        """Test with_retry with custom retry settings."""
        mock_func = Mock(side_effect=[URLError("fail"), "success"])

        result = with_retry(
            mock_func,
            "test_op",
            max_retries=3,
            initial_delay=0.01,
            verbose=False,
        )

        assert result == "success"


class TestExceptions:
    """Tests for custom exception classes."""

    def test_download_error(self) -> None:
        """Test DownloadError exception."""
        error = DownloadError("Download failed")
        assert str(error) == "Download failed"

    def test_load_error(self) -> None:
        """Test LoadError exception."""
        error = LoadError("Load failed")
        assert str(error) == "Load failed"

    def test_fatal_load_error(self) -> None:
        """Test FatalLoadError exception."""
        error = FatalLoadError("Fatal error")
        assert str(error) == "Fatal error"

    def test_rate_limit_error(self) -> None:
        """Test RateLimitError exception."""
        error = RateLimitError("Rate limited")
        assert str(error) == "Rate limited"

    def test_neo4j_transient_error(self) -> None:
        """Test Neo4jTransientError exception."""
        error = Neo4jTransientError("Deadlock detected")
        assert str(error) == "Deadlock detected"


class TestNeo4jTransientErrorDetection:
    """Tests for Neo4j transient error detection."""

    def test_is_neo4j_transient_error_deadlock(self) -> None:
        """Test detection of Neo4j deadlock error."""
        exc = Exception(
            "{neo4j_code: Neo.TransientError.Transaction.DeadlockDetected} "
            "ForsetiClient can't acquire ExclusiveLock"
        )
        assert is_neo4j_transient_error(exc) is True

    def test_is_neo4j_transient_error_lock_stopped(self) -> None:
        """Test detection of Neo4j lock client stopped error."""
        exc = Exception("Neo.TransientError.Transaction.LockClientStopped")
        assert is_neo4j_transient_error(exc) is True

    def test_is_neo4j_transient_error_terminated(self) -> None:
        """Test detection of Neo4j transaction terminated error."""
        exc = Exception("Neo.TransientError.Transaction.Terminated")
        assert is_neo4j_transient_error(exc) is True

    def test_is_neo4j_transient_error_database_unavailable(self) -> None:
        """Test detection of Neo4j database unavailable error."""
        exc = Exception("Neo.TransientError.Database.DatabaseUnavailable")
        assert is_neo4j_transient_error(exc) is True

    def test_is_neo4j_transient_error_not_transient(self) -> None:
        """Test that non-transient errors are not detected."""
        exc = Exception("Some other database error")
        assert is_neo4j_transient_error(exc) is False

    def test_is_neo4j_transient_error_by_type_name(self) -> None:
        """Test detection by exception type name."""

        class TransientError(Exception):
            pass

        exc = TransientError("some transient error")
        assert is_neo4j_transient_error(exc) is True


class TestRetryHandlerNeo4jErrors:
    """Tests for RetryHandler with Neo4j transient errors."""

    def test_execute_neo4j_deadlock_retries(self) -> None:
        """Test that Neo4j deadlock errors trigger retry."""
        handler = RetryHandler(max_retries=2, initial_delay=0.01, verbose=False)

        # Simulate a deadlock error message like the one from Neo4j
        deadlock_exc = Exception(
            "{neo4j_code: Neo.TransientError.Transaction.DeadlockDetected} "
            "ForsetiClient[transactionId=39] can't acquire ExclusiveLock"
        )
        mock_func = Mock(side_effect=[deadlock_exc, "success"])

        result = handler.execute(mock_func, "test_op")

        assert result == "success"
        assert mock_func.call_count == 2

    def test_execute_neo4j_deadlock_fails_after_max_retries(self) -> None:
        """Test that Neo4j deadlock errors fail after max retries."""
        handler = RetryHandler(max_retries=2, initial_delay=0.01, verbose=False)

        deadlock_exc = Exception(
            "{neo4j_code: Neo.TransientError.Transaction.DeadlockDetected} "
            "ForsetiClient can't acquire lock"
        )
        mock_func = Mock(side_effect=deadlock_exc)

        with pytest.raises(FatalLoadError, match="DeadlockDetected"):
            handler.execute(mock_func, "test_op")

        # Initial attempt + 2 retries = 3 calls
        assert mock_func.call_count == 3

    def test_execute_non_transient_neo4j_error_no_retry(self) -> None:
        """Test that non-transient errors are not retried."""
        handler = RetryHandler(max_retries=2, initial_delay=0.01, verbose=False)

        # A non-transient Neo4j error
        exc = ValueError("Invalid query syntax")
        mock_func = Mock(side_effect=exc)

        with pytest.raises(ValueError, match="Invalid query syntax"):
            handler.execute(mock_func, "test_op")

        # Should fail immediately, no retries
        assert mock_func.call_count == 1
