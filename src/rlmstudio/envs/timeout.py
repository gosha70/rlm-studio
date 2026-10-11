# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0

"""Timeout mechanisms for safe code execution."""

import multiprocessing as mp
import signal
from collections.abc import Callable
from typing import Any

from rlmstudio.core.errors import ExecutionError


class TimeoutError(ExecutionError):  # noqa: A001
    """Raised when code execution times out."""

    pass


class SignalTimeout:
    """
    Unix-only timeout using signal.alarm().
    Fast and efficient but only works on Unix-like systems.
    """

    def __init__(self, seconds: float):
        """
        Initialize timeout.

        Args:
            seconds: Timeout duration in seconds
        """
        self.seconds = int(seconds)
        if self.seconds <= 0:
            self.seconds = 1

    def __enter__(self) -> "SignalTimeout":
        """Start the timeout."""

        # Set up signal handler
        def timeout_handler(signum: int, frame: Any) -> None:
            raise TimeoutError(f"Code execution timed out after {self.seconds} seconds")

        # Store old handler and set new one
        self.old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Cancel the timeout."""
        signal.alarm(0)  # Cancel alarm
        signal.signal(signal.SIGALRM, self.old_handler)  # Restore old handler


class ProcessTimeout:
    """
    Cross-platform timeout using multiprocessing.
    Works on all platforms but has more overhead.
    """

    def __init__(self, seconds: float):
        """
        Initialize timeout.

        Args:
            seconds: Timeout duration in seconds
        """
        self.seconds = seconds
        self.process: mp.Process | None = None

    @staticmethod
    def _run_with_timeout(func: Callable, queue: mp.Queue, *args: Any, **kwargs: Any) -> None:
        """
        Run function in separate process and put result in queue.

        Args:
            func: Function to run
            queue: Queue to store result
            *args: Function arguments
            **kwargs: Function keyword arguments
        """
        try:
            result = func(*args, **kwargs)
            queue.put(("success", result))
        except Exception as e:
            queue.put(("error", e))

    def run(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Run function with timeout.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            TimeoutError: If execution times out
            Exception: If function raises exception
        """
        queue: mp.Queue = mp.Queue()

        # Start process
        self.process = mp.Process(
            target=self._run_with_timeout, args=(func, queue) + args, kwargs=kwargs
        )
        self.process.start()

        # Wait for completion or timeout
        self.process.join(timeout=self.seconds)

        # Check if timed out
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.0)
            if self.process.is_alive():
                self.process.kill()
            raise TimeoutError(f"Code execution timed out after {self.seconds} seconds")

        # Get result from queue
        if not queue.empty():
            status, result = queue.get()
            if status == "error":
                raise result
            return result

        # Process ended without putting result (crashed)
        raise ExecutionError("Code execution process crashed")


def create_timeout(seconds: float, use_signal: bool = True) -> Any:
    """
    Create appropriate timeout mechanism for the platform.

    Args:
        seconds: Timeout duration in seconds
        use_signal: If True and on Unix, use signal.alarm (faster)
                   If False or on Windows, use multiprocessing

    Returns:
        Timeout context manager or handler
    """
    # Check if signal.alarm is available (Unix only)
    has_signal_alarm = hasattr(signal, "SIGALRM")

    if use_signal and has_signal_alarm:
        return SignalTimeout(seconds)
    else:
        return ProcessTimeout(seconds)


def with_timeout(seconds: float, use_signal: bool = True) -> Callable:
    """
    Decorator to add timeout to a function.

    Args:
        seconds: Timeout duration in seconds
        use_signal: Whether to use signal-based timeout on Unix

    Returns:
        Decorated function with timeout

    Example:
        @with_timeout(5.0)
        def long_running_function():
            # code here
            pass
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            timeout = create_timeout(seconds, use_signal)

            if isinstance(timeout, SignalTimeout):
                # Use context manager for signal timeout
                with timeout:
                    return func(*args, **kwargs)
            else:
                # Use run method for process timeout
                return timeout.run(func, *args, **kwargs)

        return wrapper

    return decorator
