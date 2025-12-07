"""Tests for loader retry module."""

from __future__ import annotations

from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

import pytest

from nerve.loader.retry import (
    DownloadError,
    FatalLoadError,
    LoadError,
    RateLimitError,
    RetryHandler,
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
