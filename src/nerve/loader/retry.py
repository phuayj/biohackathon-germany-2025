"""Retry handler with exponential backoff.

This module provides retry logic for transient failures during
download and load operations.
"""

from __future__ import annotations

import time
from http.client import IncompleteRead
from typing import Callable, TypeVar
from urllib.error import HTTPError, URLError

# Default retry settings
DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_DELAY = 2.0  # seconds
MAX_DELAY_CAP = 60.0  # cap exponential backoff


class DownloadError(Exception):
    """Raised when a download operation fails after all retries."""

    pass


class LoadError(Exception):
    """Raised when a load operation fails after all retries."""

    pass


class FatalLoadError(Exception):
    """Raised when a load operation fails in a non-recoverable way."""

    pass


class RateLimitError(Exception):
    """Raised when an API returns a rate limit error."""

    pass


# Errors that should trigger retry
RETRYABLE_ERRORS = (
    URLError,
    HTTPError,
    ConnectionError,
    TimeoutError,
    RateLimitError,
    IncompleteRead,
    OSError,  # Includes various network-related errors
)


T = TypeVar("T")


class RetryHandler:
    """Handles retries with exponential backoff."""

    def __init__(
        self,
        max_retries: int = DEFAULT_MAX_RETRIES,
        initial_delay: float = DEFAULT_INITIAL_DELAY,
        max_delay: float = MAX_DELAY_CAP,
        verbose: bool = True,
    ) -> None:
        """Initialize retry handler.

        Args:
            max_retries: Maximum number of retry attempts.
            initial_delay: Initial delay between retries in seconds.
            max_delay: Maximum delay cap for exponential backoff.
            verbose: Whether to print retry messages.
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.verbose = verbose

    def execute(
        self,
        func: Callable[[], T],
        operation_name: str,
        retryable_errors: tuple[type[Exception], ...] = RETRYABLE_ERRORS,
    ) -> T:
        """Execute function with retry logic.

        Args:
            func: Callable to execute.
            operation_name: Name for logging purposes.
            retryable_errors: Tuple of exception types that trigger retry.

        Returns:
            Result from func()

        Raises:
            FatalLoadError: After all retries exhausted.
        """
        delay = self.initial_delay
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                return func()
            except retryable_errors as e:
                last_error = e
                if attempt == self.max_retries:
                    break

                # Check for HTTP 4xx errors (client errors) - don't retry
                if isinstance(e, HTTPError) and 400 <= e.code < 500:
                    if e.code != 429:  # 429 = Too Many Requests, should retry
                        raise FatalLoadError(f"{operation_name} failed: HTTP {e.code}") from e

                if self.verbose:
                    print(
                        f"  ⚠ {operation_name} failed (attempt {attempt + 1}/{self.max_retries + 1}), "
                        f"retrying in {delay:.1f}s..."
                    )
                time.sleep(delay)
                delay = min(delay * 2, self.max_delay)

        raise FatalLoadError(
            f"{operation_name} failed after {self.max_retries + 1} attempts: {last_error}"
        ) from last_error


def with_retry(
    func: Callable[[], T],
    operation_name: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    max_delay: float = MAX_DELAY_CAP,
    verbose: bool = True,
) -> T:
    """Convenience function for executing with retry.

    Args:
        func: Callable to execute.
        operation_name: Name for logging purposes.
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay cap for exponential backoff.
        verbose: Whether to print retry messages.

    Returns:
        Result from func()

    Raises:
        FatalLoadError: After all retries exhausted.
    """
    handler = RetryHandler(
        max_retries=max_retries,
        initial_delay=initial_delay,
        max_delay=max_delay,
        verbose=verbose,
    )
    return handler.execute(func, operation_name)
