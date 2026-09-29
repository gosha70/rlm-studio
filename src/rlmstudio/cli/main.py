"""``rlm-studio`` console-script entry point.

The CLI is intentionally tiny: ``argparse``, no third-party CLI library.
Each subcommand lives in its own helper function so the dispatcher stays
flat.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
import webbrowser
from collections.abc import Sequence

from rlmstudio import __version__
from rlmstudio.branding import (
    CLI_NAME,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DIST_NAME,
    PACKAGE_NAME,
    PRODUCT_NAME,
    env,
    env_name,
)
from rlmstudio.ui_bundle import get_ui_directory

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=CLI_NAME,
        description=f"{PRODUCT_NAME} — a workbench for Recursive Language Models.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # rlm-studio version
    subparsers.add_parser(
        "version",
        help=f"Print the installed {PRODUCT_NAME} version and exit.",
    )

    # rlm-studio studio
    studio = subparsers.add_parser(
        "studio",
        help="Start RLM Studio (API + bundled web UI on a single port).",
        description=(
            "Start RLM Studio: the FastAPI backend with the bundled "
            "Next.js UI mounted at /studio. Opens your default browser "
            "to the UI unless --no-browser is passed."
        ),
    )
    studio.add_argument(
        "--host",
        default=env("HOST", DEFAULT_HOST),
        help=f"Bind host (default: {DEFAULT_HOST}, env: {env_name('HOST')})",
    )
    studio.add_argument(
        "--port",
        type=int,
        default=int(env("PORT", str(DEFAULT_PORT)) or DEFAULT_PORT),
        help=f"Bind port (default: {DEFAULT_PORT}, env: {env_name('PORT')})",
    )
    studio.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the browser automatically.",
    )
    studio.add_argument(
        "--browser-delay",
        type=float,
        default=1.5,
        help=(
            "Seconds to wait after server start before opening the "
            "browser (default: 1.5). Increase if the browser opens "
            "before the server is ready."
        ),
    )
    studio.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on source changes (development only).",
    )

    # rlm-studio bench
    bench = subparsers.add_parser(
        "bench",
        help="Run the reproducible benchmark set and regenerate BENCHMARKS.md.",
        description=(
            "Run a YAML benchmark across providers × engines through the same "
            "code path as the Compare page, score every cell (match + LLM judge), "
            "write results.json / results.md, and optionally regenerate the "
            "results section of BENCHMARKS.md."
        ),
    )
    bench.add_argument(
        "--config", required=True, help="Benchmark YAML (e.g. benchmarks/longdoc-v1.yaml)."
    )
    bench.add_argument(
        "--providers",
        required=True,
        help='Comma-separated "backend/model" specs, e.g. openai/gpt-4o-mini,ollama/qwen3:8b.',
    )
    bench.add_argument(
        "--engines",
        default="direct,rag,rlm",
        help="Comma-separated slot modes: direct, rag, rlm, rlm_official (default: direct,rag,rlm).",
    )
    bench.add_argument(
        "--judge",
        default=None,
        help='Judge "backend/model" spec; omit for the accuracy proxy only.',
    )
    bench.add_argument(
        "--out", required=True, help="Output directory for results.json and results.md."
    )
    bench.add_argument(
        "--page", default=None, help="BENCHMARKS.md to regenerate between its bench markers."
    )
    bench.add_argument("--reps", type=int, default=1, help="Repetitions per case (default: 1).")
    bench.add_argument(
        "--limit", type=int, default=None, help="Run only the first N materialised cases."
    )
    bench.add_argument("--cases", default=None, help="Comma-separated case ids to run.")
    bench.add_argument(
        "--fetch", action="store_true", help="Download public texts into the corpus directory."
    )
    bench.add_argument(
        "--corpus-dir",
        default=None,
        help="Where public texts live (default: <config dir>/corpus).",
    )
    bench.add_argument(
        "--dry-run", action="store_true", help="Use offline fakes for every adapter (CI smoke)."
    )
    bench.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature (default: 0)."
    )
    bench.add_argument("--api-key", default=None, help="API key applied to every slot.")
    bench.add_argument("--api-base", default=None, help="API base URL applied to every slot.")
    bench.add_argument(
        "--timeout", type=float, default=None, help="Per-request timeout in seconds."
    )
    bench.add_argument(
        "--sandbox",
        default=None,
        help="Sandbox for the rlm_official engine: restricted (in-process) or docker.",
    )
    bench.add_argument(
        "--note", default=None, help="Free-text run note (hardware, model versions)."
    )

    return parser


def _cmd_version() -> int:
    print(f"{CLI_NAME} {__version__}")
    return 0


_WILDCARD_BIND_HOSTS = frozenset({"0.0.0.0", "::", "*", ""})


def _browse_host(host: str) -> str:
    """Substitute a browseable loopback for wildcard bind hosts.

    The actual uvicorn bind still uses ``host`` verbatim; this rewrites
    only the URL that is printed and opened in the browser. Browsers do
    not consistently navigate to ``0.0.0.0`` or ``[::]`` — those are
    bind targets, not addressable hosts — so the advertised
    ``rlm-studio studio --host 0.0.0.0`` workflow needs a navigable URL
    even though the bind is correct as written.
    """
    return "127.0.0.1" if host in _WILDCARD_BIND_HOSTS else host


def _cmd_studio(args: argparse.Namespace) -> int:
    import uvicorn

    ui_dir = get_ui_directory()
    if ui_dir is None:
        print(
            f"{PRODUCT_NAME}: no bundled UI was found.\n"
            "\n"
            f"  Expected at: {PACKAGE_NAME}/_ui/index.html\n"
            "\n"
            f"This usually means you installed {DIST_NAME} from a source\n"
            "checkout without building the frontend. Two options:\n"
            "\n"
            "  1. Install the published wheel:\n"
            f"       pip install --upgrade {DIST_NAME}\n"
            "\n"
            "  2. Build the frontend locally:\n"
            "       cd frontend\n"
            "       npm install && npm run build:bundle\n"
            "       # then copy frontend/out → src/rlmstudio/_ui/\n"
            "\n"
            "  3. Or run the dev stack with the API and UI as separate\n"
            "     processes:\n"
            f"       python -m {PACKAGE_NAME}.server        # terminal 1\n"
            "       cd frontend && npm run dev         # terminal 2\n",
            file=sys.stderr,
        )
        return 2

    browse_host = _browse_host(args.host)
    url = f"http://{browse_host}:{args.port}/studio"
    print(f"RLM Studio: {url}")
    print(f"  API:        http://{browse_host}:{args.port}/api")
    print(f"  Health:     http://{browse_host}:{args.port}/health")
    print()

    if not args.no_browser:
        # Open the browser on a delay so the server has time to bind.
        # We do this in a daemon thread so it does not block uvicorn.run().
        def _open_browser() -> None:
            time.sleep(args.browser_delay)
            try:
                webbrowser.open(url)
            except Exception as exc:  # pragma: no cover - depends on host env
                logger.warning("Could not open browser: %s", exc)

        threading.Thread(target=_open_browser, daemon=True).start()

    uvicorn.run(
        "rlmstudio.server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def _csv(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _cmd_bench(args: argparse.Namespace) -> int:
    from pathlib import Path

    from rlmstudio.api import MATRIX_MODES, build_llm_adapter
    from rlmstudio.benchmark.corpus import materialise
    from rlmstudio.benchmark.dataset import load_dataset
    from rlmstudio.benchmark.fakes import dry_run_judge_llm, dry_run_slot_builder
    from rlmstudio.benchmark.matrix_runner import MatrixBenchmarkRunner, SlotOutcome
    from rlmstudio.benchmark.report import (
        RESULTS_METADATA_NOTE,
        MatrixBenchmarkReport,
        regenerate_page,
    )
    from rlmstudio.benchmark.scoring import JudgeScorer

    providers = _csv(args.providers)
    engines = _csv(args.engines)
    unknown = [e for e in engines if e not in MATRIX_MODES]
    if unknown:
        print(
            f"{CLI_NAME} bench: unknown engine(s) {', '.join(unknown)}; "
            f"valid: {', '.join(sorted(MATRIX_MODES))}",
            file=sys.stderr,
        )
        return 2

    config_path = Path(args.config)
    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else config_path.parent / "corpus"
    dataset = load_dataset(str(config_path))
    materialised = materialise(
        dataset, base_dir=config_path.parent, corpus_dir=corpus_dir, fetch=args.fetch
    )
    for case_id, reason in materialised.skipped.items():
        print(f"  skip   {case_id}: {reason}")

    judge = None
    if args.judge:
        judge_llm = (
            dry_run_judge_llm()
            if args.dry_run
            else build_llm_adapter(
                args.judge, api_key=args.api_key, api_base=args.api_base, temperature=0.0
            )
        )
        judge = JudgeScorer(judge_llm, model=args.judge)

    # Kept so a run that dies part-way still leaves the cells it paid for on
    # disk: a real run is hours of provider calls, and losing all of it to one
    # unexpected error at the end would be the most expensive kind of bug.
    completed: list[SlotOutcome] = []

    def _progress(outcome: SlotOutcome) -> None:
        completed.append(outcome)
        marker = "✓" if outcome.success else "✗"
        judge_note = f" · judge failed: {outcome.judge_error}" if outcome.judge_error else ""
        print(
            f"  {marker} {outcome.case_id} · {outcome.provider}/{outcome.model} · {outcome.engine}"
            f" · {outcome.outcome} · {outcome.elapsed_seconds:.1f}s{judge_note}"
        )

    runner = MatrixBenchmarkRunner(
        providers=providers,
        engines=engines,
        slot_builder=dry_run_slot_builder if args.dry_run else None,
        judge=judge,
        reps=args.reps,
        temperature=args.temperature,
        api_key=args.api_key,
        api_base=args.api_base,
        timeout=args.timeout,
        sandbox_type=args.sandbox,
        on_slot_complete=_progress,
    )
    out_dir = Path(args.out)
    try:
        results = runner.run(
            dataset,
            case_ids=_csv(args.cases) or None,
            limit=args.limit,
            skipped=materialised.skipped,
        )
    except BaseException as exc:  # including KeyboardInterrupt: save, then re-raise
        partial = runner.partial_results(dataset, completed, skipped=materialised.skipped)
        partial.metadata.update(
            {
                RESULTS_METADATA_NOTE: args.note,
                "dry_run": args.dry_run,
                "config": str(config_path),
                "incomplete": f"{type(exc).__name__}: {exc}",
            }
        )
        path = out_dir / "results.partial.json"
        MatrixBenchmarkReport(partial).save_json(path)
        print(
            f"\n{CLI_NAME} bench: run stopped after {len(completed)} cell(s); "
            f"wrote what completed to {path}",
            file=sys.stderr,
        )
        raise
    results.metadata.update(
        {RESULTS_METADATA_NOTE: args.note, "dry_run": args.dry_run, "config": str(config_path)}
    )

    report = MatrixBenchmarkReport(results)
    report.save_json(out_dir / "results.json")
    report.save_markdown(out_dir / "results.md")
    print()
    print(report.to_markdown())
    print(f"Wrote {out_dir / 'results.json'} and {out_dir / 'results.md'}")
    if args.page:
        regenerate_page(args.page, report.to_markdown())
        print(f"Regenerated {args.page}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "version":
        return _cmd_version()
    if args.command == "studio":
        return _cmd_studio(args)
    if args.command == "bench":
        try:
            return _cmd_bench(args)
        except (ValueError, OSError, RuntimeError) as exc:
            # RuntimeError is what the LLM adapter raises for a provider that
            # rate-limited, timed out or refused the key: a clean message, not a
            # traceback.  Anything already completed has been saved by now.
            print(f"{CLI_NAME} bench: {exc}", file=sys.stderr)
            return 1

    parser.error(f"Unknown command: {args.command!r}")
    return 2  # unreachable; parser.error exits.


if __name__ == "__main__":
    raise SystemExit(main())
