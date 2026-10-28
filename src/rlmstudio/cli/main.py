"""``rlm-studio`` console-script entry point.

The CLI is intentionally tiny: ``argparse``, no third-party CLI library.
Each subcommand lives in its own helper function so the dispatcher stays
flat.
"""

from __future__ import annotations

import argparse
import errno
import logging
import socket
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


def _bind_error(host: str, port: int) -> str | None:
    """Return a user-facing message if ``host:port`` cannot be bound.

    uvicorn reports a bind failure only *after* the startup banner has
    been printed, so without this pre-flight the user is shown a URL
    that is served by whatever already owns the port — most often
    another RLM Studio. Returning the message here lets the caller fail
    before advertising an address it does not own.

    This is advisory: the port can still be taken between the probe and
    uvicorn's own bind. That race turns a friendly message back into
    uvicorn's traceback, which is the pre-existing behaviour, not a
    regression.
    """
    # Pick the family exactly the way uvicorn does, then resolve *within*
    # it. Both halves matter:
    #
    #   * Choosing by hand is what uvicorn does (``AF_INET`` unless the
    #     host contains a colon, see uvicorn/config.py), and letting
    #     getaddrinfo order the families instead silently disagrees with
    #     it: on macOS ``localhost`` resolves to ``::1`` first and ``''``
    #     to ``::`` first, while uvicorn binds ``127.0.0.1`` and
    #     ``0.0.0.0``. A probe that tested the wrong family would refuse
    #     a start that works — something only listening on ``[::1]``,
    #     which Node and Vite dev servers commonly do, would look like a
    #     clash on ``localhost``. Refusing a good start is worse than the
    #     race this pre-flight already accepts.
    #
    #   * Resolving is still required, because a name has to become an
    #     address and an IPv6 sockaddr carries flowinfo and scope-id that
    #     a hand-built 2-tuple does not.
    family = socket.AF_INET6 if (host and ":" in host) else socket.AF_INET
    try:
        candidates = socket.getaddrinfo(
            host, port, family=family, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
        )
    except socket.gaierror:
        return (
            f"{PRODUCT_NAME} could not start: {host!r} is not a host name "
            "this machine can resolve.\n"
            "\n"
            "  Check the spelling, or bind an address directly:\n"
            f"       {CLI_NAME} studio --host {DEFAULT_HOST} --port {port}"
        )

    resolved_family, socktype, proto, _canonname, sockaddr = candidates[0]
    # Name the address actually being bound. ``--host ''`` is a supported
    # wildcard, and echoing the raw host there leaves a blank where the
    # address should be ("port 8000 on  is already in use"); the resolved
    # form also tells a `localhost` user which family the clash is on.
    shown_host = str(sockaddr[0]) or host
    probe = socket.socket(resolved_family, socktype, proto)
    try:
        # Mirror uvicorn's own socket options so the probe accepts and
        # rejects exactly what the real bind will.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(sockaddr)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            return (
                f"{PRODUCT_NAME} could not start: port {port} on {shown_host} is "
                "already in use.\n"
                "\n"
                "Something else is listening there — often another copy of "
                f"{PRODUCT_NAME} from an earlier terminal.\n"
                "\n"
                "  Use a different port:\n"
                f"       {CLI_NAME} studio --port {port + 1}\n"
                "\n"
                "  Or find what is holding it:\n"
                f"       lsof -nP -iTCP:{port} -sTCP:LISTEN        # macOS / Linux\n"
                f"       netstat -ano | findstr :{port}            # Windows\n"
                "\n"
                f"  The port can also be set with {env_name('PORT')}."
            )
        if exc.errno in (errno.EACCES, errno.EPERM):
            return (
                f"{PRODUCT_NAME} could not start: no permission to bind port "
                f"{port} on {shown_host}.\n"
                "\n"
                "Ports below 1024 are reserved for privileged processes.\n"
                "\n"
                "  Use a port above 1024:\n"
                f"       {CLI_NAME} studio --port {DEFAULT_PORT}"
            )
        if exc.errno in (errno.EADDRNOTAVAIL, errno.EINVAL):
            return (
                f"{PRODUCT_NAME} could not start: {shown_host} is not an address "
                "this machine can bind.\n"
                "\n"
                "  Bind all interfaces instead:\n"
                f"       {CLI_NAME} studio --host 0.0.0.0 --port {port}"
            )
        raise
    finally:
        probe.close()
    return None


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

    # Check the bind *before* printing any URL: advertising an address we
    # do not own sends the user to someone else's process.
    bind_error = _bind_error(args.host, args.port)
    if bind_error is not None:
        print(bind_error, file=sys.stderr)
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
