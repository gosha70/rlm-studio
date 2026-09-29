"""Root pytest conftest — ``--runslow`` gate for heavy tests.

Some tests (notably the subprocess-sandbox fresh-interpreter entrypoint
tests) spawn multiple fresh Python interpreters per case and take 5–15
seconds each.  They are essential for CI but painful in local iteration.

This module wires the already-declared ``slow`` marker in
``pyproject.toml`` to a ``--runslow`` command-line gate using the
standard pytest pattern:

- Default behavior: tests marked ``@pytest.mark.slow`` are **skipped**
  with a clear reason message.  Run all other tests fast.
- ``pytest --runslow``: include slow tests.  CI passes this flag.

This is NOT a CI-only mechanism — contributors can ``pytest --runslow``
locally when they specifically want the full suite.  But ``pytest`` with
no flags stays fast.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--runslow`` CLI flag."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.slow (spawn-heavy, 5-15s each).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip slow-marked tests unless ``--runslow`` was passed."""
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="slow test — pass --runslow to include (pytest --runslow)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(autouse=True)
def _working_directory_is_restored() -> Iterator[None]:
    """Fail the test that leaves the process chdir'd somewhere else.

    The working directory is process-global, so a test that changes it — or
    that abandons a thread inside code which changes it, as the official RLM
    engine's in-process REPL does — silently breaks any later test in the same
    worker that spawns a subprocess: the child inherits a directory that may
    already be deleted and dies on its first ``os.getcwd()``.  That failure
    surfaces far from its cause, so it is caught here instead.
    """
    import os

    before = os.getcwd()
    yield
    after = os.getcwd()
    if after != before:
        os.chdir(before)
        pytest.fail(f"test left the working directory at {after!r} instead of {before!r}")


def pytest_configure(config: pytest.Config) -> None:
    """Isolate every test process from the developer's real state directory.

    ``rlmstudio.branding.state_dir()`` honours ``RLM_STUDIO_DIR``; pointing
    it at a per-session temp dir before any ``rlmstudio`` module is imported
    guarantees the suite never reads, writes, or migrates ``~/.rlm-studio``
    / ``~/.rlmkit``.  Tests that need the *default* resolution monkeypatch
    the variable away themselves.
    """
    import os
    import tempfile

    os.environ.setdefault("RLM_STUDIO_DIR", tempfile.mkdtemp(prefix="rlm-studio-test-state-"))
