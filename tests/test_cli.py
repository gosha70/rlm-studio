"""Smoke tests for the ``rlm-studio`` console script.

These tests deliberately do NOT boot a real uvicorn server. They cover:

* The CLI module imports cleanly.
* ``rlmstudio version`` exits 0 and prints a version string.
* ``rlm-studio studio --help`` exits 0 and lists the documented flags.
* The bundled-UI helper returns the right value for the current source
  tree state (``None`` when ``_ui/`` is empty in the dev checkout).
* The FastAPI app still constructs without a bundled UI (regression
  guard for the conditional mount added in ``server/app.py``).

The end-to-end "``rlm-studio studio`` actually serves the UI" test belongs
in the manual release plan, not in pytest. See
``doc_internal/release/steps/01-pre-cut-gates.md`` § "One-click smoke
test" once that section lands.
"""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout

import pytest

from rlmstudio.cli import main as cli_main_module
from rlmstudio.cli.main import _browse_host, main
from rlmstudio.ui_bundle import get_ui_directory, is_ui_bundled


def test_cli_module_imports() -> None:
    """The CLI module must be importable without side effects."""
    assert callable(cli_main_module.main)


def test_version_command_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["version"])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.startswith("rlm-studio ")


def test_studio_help_exits_zero() -> None:
    """`rlm-studio studio --help` should print usage and exit 0.

    argparse exits via SystemExit, not return; catch it explicitly.
    """
    buf = io.StringIO()
    with pytest.raises(SystemExit) as exc_info, redirect_stdout(buf):
        main(["studio", "--help"])
    assert exc_info.value.code == 0
    help_text = buf.getvalue()
    assert "--host" in help_text
    assert "--port" in help_text
    assert "--no-browser" in help_text
    assert "--reload" in help_text


def test_top_level_help_exits_zero() -> None:
    buf = io.StringIO()
    with pytest.raises(SystemExit) as exc_info, redirect_stdout(buf):
        main(["--help"])
    assert exc_info.value.code == 0
    help_text = buf.getvalue()
    assert "studio" in help_text
    assert "version" in help_text


def test_unknown_command_exits_nonzero() -> None:
    """argparse rejects unknown subcommands with exit code 2."""
    buf = io.StringIO()
    with pytest.raises(SystemExit) as exc_info, redirect_stderr(buf):
        main(["definitely-not-a-real-subcommand"])
    assert exc_info.value.code != 0


def test_studio_without_bundled_ui_prints_helpful_message(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When the bundled UI is missing, `rlm-studio studio` must:

    * exit non-zero (not crash uvicorn)
    * print actionable guidance pointing at the dev workflow
    * NOT attempt to start uvicorn
    """
    # Force the UI lookup to return None regardless of the actual state.
    monkeypatch.setattr("rlmstudio.cli.main.get_ui_directory", lambda: None)

    # Sentinel that fails the test if uvicorn.run is invoked.
    def _fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("uvicorn.run must not be called when UI is missing")

    monkeypatch.setattr("uvicorn.run", _fail)

    rc = main(["studio", "--no-browser", "--port", "0"])
    captured = capsys.readouterr()

    assert rc != 0
    # Match exact-case to keep the assertion honest; the CLI prints
    # "RLM Studio: no bundled UI was found." and the dev workflow hints.
    assert "no bundled UI" in captured.err
    assert "npm run build:bundle" in captured.err
    assert "python -m rlmstudio.server" in captured.err


def test_ui_bundle_helpers_consistent() -> None:
    """`is_ui_bundled()` must agree with `get_ui_directory() is not None`."""
    assert is_ui_bundled() is (get_ui_directory() is not None)


def test_server_app_constructs_without_bundled_ui() -> None:
    """Regression guard: the FastAPI app must import cleanly when
    ``src/rlmstudio/_ui/`` is empty (no ``index.html``), which is the
    state of every source checkout that has not run the frontend build.

    If this test fails, the conditional mount in
    ``rlmstudio.server.app.create_app`` is regressed and the API-only
    workflow (`python -m rlmstudio.server` + `npm run dev` on :3000) is
    broken.
    """
    from rlmstudio.server.app import app  # noqa: F401 — import is the test.


@pytest.mark.parametrize(
    ("bind_host", "expected"),
    [
        ("0.0.0.0", "127.0.0.1"),
        ("::", "127.0.0.1"),
        ("*", "127.0.0.1"),
        ("", "127.0.0.1"),
        ("127.0.0.1", "127.0.0.1"),
        ("localhost", "localhost"),
        ("192.168.1.100", "192.168.1.100"),
        ("studio.internal.example", "studio.internal.example"),
    ],
)
def test_browse_host_maps_wildcards_to_loopback(bind_host: str, expected: str) -> None:
    """`_browse_host` substitutes a navigable loopback for wildcard
    bind targets and preserves anything else verbatim.

    The actual uvicorn bind continues to use the raw host so
    ``rlm-studio studio --host 0.0.0.0`` still binds the wildcard; this
    rewrite only applies to the URL we print and auto-open in the
    browser. Wildcard targets (``0.0.0.0``, ``::``) are not
    consistently browseable across platforms — Linux usually routes
    them to loopback, macOS is inconsistent, Windows tends to fail —
    so we hand the browser a loopback URL.
    """
    assert _browse_host(bind_host) == expected


# ---------------------------------------------------------------------------
# Bind defaults come from rlmstudio.branding, and the env vars win
# ---------------------------------------------------------------------------


def test_bind_defaults_come_from_branding_constants() -> None:
    """The parser must not carry its own copies of the host/port literals."""
    from rlmstudio.branding import DEFAULT_HOST, DEFAULT_PORT
    from rlmstudio.cli.main import _build_parser

    args = _build_parser().parse_args(["studio"])
    assert args.host == DEFAULT_HOST
    assert args.port == DEFAULT_PORT


def test_env_overrides_bind_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """RLM_STUDIO_HOST / _PORT drive the defaults — this is what the Docker
    image relies on now that its CMD no longer hardcodes --port."""
    import importlib

    monkeypatch.setenv("RLM_STUDIO_HOST", "0.0.0.0")
    monkeypatch.setenv("RLM_STUDIO_PORT", "9000")
    main_mod = importlib.import_module("rlmstudio.cli.main")

    args = main_mod._build_parser().parse_args(["studio"])
    assert args.host == "0.0.0.0"
    assert args.port == 9000


def test_explicit_flags_still_beat_the_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RLM_STUDIO_PORT", "9000")
    from rlmstudio.cli.main import _build_parser

    args = _build_parser().parse_args(["studio", "--port", "7777"])
    assert args.port == 7777


def test_bind_error_is_none_for_a_free_port() -> None:
    """A port nothing is listening on must probe clean.

    Port 0 asks the kernel for an ephemeral port, which is always
    available — this pins the happy path so the pre-flight check cannot
    start rejecting ordinary startups.
    """
    from rlmstudio.cli.main import _bind_error

    assert _bind_error("127.0.0.1", 0) is None


def test_bind_error_reports_a_port_already_in_use() -> None:
    """A held port must produce actionable text, not a bare errno.

    uvicorn reports the bind failure only after the startup banner has
    printed, which sends the user to a URL served by whatever already
    owns the port. The message must name the port and offer a way out.
    """
    import socket

    from rlmstudio.cli.main import _bind_error

    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        holder.bind(("127.0.0.1", 0))
        holder.listen(1)
        held_port = holder.getsockname()[1]

        message = _bind_error("127.0.0.1", held_port)
    finally:
        holder.close()

    assert message is not None
    assert str(held_port) in message
    assert "already in use" in message
    assert "--port" in message


def test_studio_refuses_to_print_a_url_it_does_not_own(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """On a port clash `studio` must exit non-zero, never start uvicorn,
    and never advertise the URL — printing one sends the user to the
    process that already holds the port."""
    monkeypatch.setattr("rlmstudio.cli.main.get_ui_directory", lambda: "/tmp/not-a-real-ui")
    monkeypatch.setattr(
        "rlmstudio.cli.main._bind_error",
        lambda _host, _port: "port 8000 already in use",
    )

    def _fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("uvicorn.run must not be called on a bind failure")

    monkeypatch.setattr("uvicorn.run", _fail)

    rc = main(["studio", "--no-browser"])
    captured = capsys.readouterr()

    assert rc != 0
    assert "already in use" in captured.err
    assert "/studio" not in captured.out


def test_bind_error_handles_ipv6_hosts() -> None:
    """`::` is in _WILDCARD_BIND_HOSTS, so `--host ::` is a supported
    invocation and the probe must not assume IPv4.

    A hardcoded AF_INET socket raises gaierror on an IPv6 literal, which
    no handler catches — turning a pre-flight meant to replace
    tracebacks into one that produces them.
    """
    from rlmstudio.cli.main import _bind_error

    assert _bind_error("::", 0) is None
    assert _bind_error("::1", 0) is None


def test_bind_error_detects_a_held_ipv6_port() -> None:
    """Resolving the family must not cost the clash detection the probe
    exists for."""
    import socket

    from rlmstudio.cli.main import _bind_error

    holder = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    try:
        holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        holder.bind(("::1", 0))
        holder.listen(1)
        held_port = holder.getsockname()[1]

        message = _bind_error("::1", held_port)
    finally:
        holder.close()

    assert message is not None
    assert "already in use" in message
    assert str(held_port) in message


def test_bind_error_explains_an_unresolvable_host() -> None:
    """An unresolvable host must produce guidance, not a gaierror
    traceback — the whole point of the pre-flight."""
    from rlmstudio.cli.main import _bind_error

    message = _bind_error("no-such-host.invalid", 8799)

    assert message is not None
    assert "resolve" in message
    assert "--host" in message


def test_bind_error_matches_uvicorn_family_for_localhost() -> None:
    """The probe must not refuse a start uvicorn would complete.

    uvicorn picks the family by hand — AF_INET unless the host contains
    a colon — and binds a plain 2-tuple. Letting getaddrinfo order the
    families instead disagrees with it: on macOS `localhost` resolves to
    `::1` first, so a probe holding only `[::1]` would report a clash
    while uvicorn binds `127.0.0.1` happily. Node and Vite dev servers
    commonly listen on `[::1]` alone, so this is an ordinary situation,
    and refusing a good start is worse than the race the probe accepts.
    """
    import socket

    from rlmstudio.cli.main import _bind_error

    holder = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    try:
        holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        holder.bind(("::1", 0))
        holder.listen(1)
        held_port = holder.getsockname()[1]

        # Only [::1]:held_port is taken. uvicorn would bind
        # 127.0.0.1:held_port, so the probe must allow it.
        assert _bind_error("localhost", held_port) is None
        # The same port on the family that IS held must still be caught.
        assert _bind_error("::1", held_port) is not None
    finally:
        holder.close()


def test_bind_error_detects_a_clash_on_the_family_uvicorn_uses() -> None:
    """The converse of the test above: an IPv4 holder must be found
    through `localhost`, which is the family uvicorn binds."""
    import socket

    from rlmstudio.cli.main import _bind_error

    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        holder.bind(("127.0.0.1", 0))
        holder.listen(1)
        held_port = holder.getsockname()[1]

        message = _bind_error("localhost", held_port)
    finally:
        holder.close()

    assert message is not None
    assert "already in use" in message


def test_bind_error_accepts_the_empty_host() -> None:
    """`''` is in _WILDCARD_BIND_HOSTS. uvicorn treats a falsy host as
    AF_INET; getaddrinfo would otherwise order `::` first on macOS."""
    from rlmstudio.cli.main import _bind_error

    assert _bind_error("", 0) is None


def test_bind_error_names_an_address_for_the_wildcard_host() -> None:
    """`--host ''` is a supported wildcard. Echoing the raw host leaves a
    blank where the address belongs ("port 8000 on  is already in use"),
    so the message must name the resolved address instead."""
    import socket

    from rlmstudio.cli.main import _bind_error

    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        holder.bind(("0.0.0.0", 0))
        holder.listen(1)
        held_port = holder.getsockname()[1]

        message = _bind_error("", held_port)
    finally:
        holder.close()

    assert message is not None
    assert "0.0.0.0" in message
    assert "on  is" not in message


def test_bind_error_passes_none_not_empty_string_to_getaddrinfo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`''` must reach getaddrinfo as None.

    `''` is one of _WILDCARD_BIND_HOSTS and uvicorn passes it to
    `socket.bind`, which reads it as the any-address. getaddrinfo does
    not agree across platforms: macOS resolves `''` to the wildcard,
    glibc raises gaierror. So `--host ''` worked on macOS and refused to
    start on Linux, and the tests written on macOS passed while CI
    failed.

    Asserting on the argument rather than the result makes this
    platform-independent: it fails on macOS too if the normalisation is
    dropped.
    """
    import socket as socket_module

    from rlmstudio.cli import main as cli_main

    seen: list[object] = []
    real = socket_module.getaddrinfo

    def _spy(host, port, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(host)
        return real(host, port, **kwargs)

    monkeypatch.setattr(cli_main.socket, "getaddrinfo", _spy)

    cli_main._bind_error("", 0)

    assert seen, "getaddrinfo was not called"
    assert seen[0] is None, f"empty host reached getaddrinfo as {seen[0]!r}, not None"
