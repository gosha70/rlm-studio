"""FastAPI application for RLMKit API server."""

from __future__ import annotations

import logging
import os as _os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from rlmstudio.branding import PRODUCT_NAME, migrate_legacy_state
from rlmstudio.server.dependencies import get_state
from rlmstudio.server.models import HealthResponse
from rlmstudio.server.routes import (
    chat,
    chat_providers,
    compare_matrix,
    config,
    diagnostics,
    docs,
    engines,
    evaluations,
    files,
    llm_providers,
    metrics,
    profiles,
    prompts,
    providers,
    replays,
    sessions,
    traces,
)
from rlmstudio.ui_bundle import get_ui_directory

# Snapshot keys already in the real process environment *before* load_dotenv.
# This lets us distinguish user-set shell variables (must not be overridden)
# from .env-loaded values (may be overridden by a newer SecretStore entry).
_real_env_keys: frozenset[str] = frozenset(_os.environ.keys())

# Load .env file (backward-compat: keys saved before the SecretStore migration)
_env_path = Path(".env")
load_dotenv(_env_path)


# Reload keys persisted via SecretStore (keyring / JSON file) into os.environ
# so all server read-paths (os.environ.get) work after restart.
# SecretStore wins over .env; real process-env vars are never overridden.
def _reload_stored_api_keys() -> None:
    from rlmstudio.ui.data.providers_catalog import PROVIDERS
    from rlmstudio.ui.services.secret_store import FileSecretStore, KeyringSecretStore

    file_store = FileSecretStore()
    keyring_store = KeyringSecretStore() if KeyringSecretStore.is_available() else None

    for p in PROVIDERS:
        if not p.env_var:
            continue
        if p.env_var in _real_env_keys:
            continue  # set in the shell before the server started — never override
        key = (keyring_store.get(p.key) if keyring_store else None) or file_store.get(p.key)
        if key:
            _os.environ[p.env_var] = key


_reload_stored_api_keys()

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    # Configure logging so all rlmstudio messages appear in the terminal
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # Set rlmstudio loggers to DEBUG so detailed traces are visible
    logging.getLogger("rlmstudio").setLevel(logging.DEBUG)

    from rlmstudio import __version__

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Startup: this is the one place the legacy state directory is
        # copied forward (explicit boot boundary — never on import, in
        # tests, or in `version`).  AppState itself constructs lazily via
        # get_state(); the background connection-test thread starts
        # automatically in AppState.__init__ if configured.
        migrate_legacy_state()
        yield
        # Shutdown: stop the scheduled-connection-testing thread before
        # process teardown.  Daemon threads die with the process
        # regardless, but the 2-second join runs BEFORE the logger shuts
        # down, so in-flight probes do not emit log lines into a
        # mid-shutdown logger.  See spec §Thread lifecycle.
        state = get_state()
        state._stop_connection_testing()

    app = FastAPI(
        title=f"{PRODUCT_NAME} API",
        version=__version__,
        description=f"{PRODUCT_NAME} — Recursive Language Model workbench API server",
        lifespan=_lifespan,
    )

    # CORS middleware for frontend dev server
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3001",
            "http://host.docker.internal:3000",
            "http://host.docker.internal:3001",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom exception handler to match ErrorResponse format from spec
    code_map = {
        400: "VALIDATION_ERROR",
        401: "UNAUTHORIZED",
        404: "NOT_FOUND",
        413: "FILE_TOO_LARGE",
        500: "INTERNAL_ERROR",
    }

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: object, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": code_map.get(exc.status_code, "ERROR"),
                    "message": str(exc.detail),
                    "details": {},
                }
            },
        )

    # Include route modules
    app.include_router(chat.router)
    app.include_router(compare_matrix.router)
    app.include_router(files.router)
    app.include_router(sessions.router)
    app.include_router(metrics.router)
    app.include_router(traces.router)
    app.include_router(providers.router)
    app.include_router(profiles.router)
    app.include_router(prompts.router)
    app.include_router(config.router)
    app.include_router(llm_providers.router)
    app.include_router(chat_providers.router)
    app.include_router(evaluations.router)
    app.include_router(diagnostics.router)
    app.include_router(docs.router)
    app.include_router(engines.router)
    app.include_router(replays.router)

    @app.get("/health")
    async def health_check() -> HealthResponse:
        state = get_state()
        from rlmstudio import __version__

        return HealthResponse(
            status="ok",
            version=__version__,
            uptime_seconds=round(time.time() - state.start_time, 1),
        )

    # ---------------------------------------------------------------------
    # Bundled Studio UI mount.
    #
    # When the Next.js static export has been built and copied into
    # ``src/rlmstudio/_ui/`` (either by the release-time build pipeline that
    # produces the wheel, or by a local developer's ``npm run build:bundle``),
    # we serve it from this same FastAPI process at ``/studio``. This is
    # what makes ``rlmstudio studio`` a single-process one-click experience.
    #
    # When the directory is not populated (which is the case in CI and
    # in any source checkout without a frontend build), we skip the
    # mount entirely and the server runs as an API-only process. The
    # frontend dev server (``npm run dev`` on :3000) remains the
    # supported flow for developers iterating on the UI.
    #
    # The mount is registered *after* all API routes so the API takes
    # precedence over the static serve for ``/api/*`` paths.
    # ---------------------------------------------------------------------
    ui_dir = get_ui_directory()
    if ui_dir is not None:
        # Root path redirects to /studio so ``rlmstudio studio`` users who
        # type the bare host:port hit the UI without needing to know the
        # mount prefix.
        @app.get("/", include_in_schema=False)
        async def _root_redirect() -> RedirectResponse:
            return RedirectResponse(url="/studio")

        # ``html=True`` makes StaticFiles serve ``index.html`` for
        # directory paths (so ``/studio/dashboard/`` resolves to
        # ``_ui/dashboard/index.html``). Next.js static export emits one
        # ``index.html`` per route which is exactly this shape.
        app.mount(
            "/studio",
            StaticFiles(directory=str(ui_dir), html=True),
            name="studio",
        )
        logger.info("RLM Studio UI mounted at /studio (from %s)", ui_dir)
    else:
        logger.info(
            "RLM Studio UI not bundled — running as API-only. "
            "Run `npm run build:bundle` in frontend/ and copy out/ → src/rlmstudio/_ui/ "
            "to enable the bundled UI."
        )

    return app


app = create_app()
