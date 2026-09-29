"""Thin wrapper: ``python benchmarks/run_benchmark.py …`` == ``rlm-studio bench …``.

See BENCHMARKS.md for the full command and ``rlm-studio bench --help`` for flags.
"""

from __future__ import annotations

import sys

from rlmstudio.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main(["bench", *sys.argv[1:]]))
