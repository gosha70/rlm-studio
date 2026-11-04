#!/usr/bin/env bash
# smoke-install.sh — run the install-path checks from the manual test plan (§1)
# without doing them by hand, one Python at a time.
#
# What it does, per Python version:
#   1. builds the bundled Studio UI (npm run build:bundle) and ships it into
#      src/rlmstudio/_ui/, and copies docs/ into src/rlmstudio/_docs/,
#      exactly as RELEASING.md describes;
#   2. builds the wheel + sdist with `uv build`;
#   3. asserts the wheel's contents (bundled UI present, no legacy rlmkit/
#      paths, correct console-script entry point);
#   4. installs that wheel with the `[studio]` extra into a throwaway venv;
#   5. checks extras hygiene — no dev toolchain, no Streamlit, leaks into a
#      user install;
#   6. boots `rlm-studio studio` on a free port with an isolated state
#      directory and asserts /health, /studio/ and a hashed asset all serve.
#
# It never touches ~/.rlm-studio: every boot runs with RLM_STUDIO_DIR pointed
# at a temp dir, which is also a live check that the override works.
#
# Usage:
#   scripts/release/smoke-install.sh                    # 3.11 3.12 3.13
#   scripts/release/smoke-install.sh 3.12               # just one
#   SKIP_BUILD=1 scripts/release/smoke-install.sh       # reuse an existing dist/
#
# Exit code is 0 only if every check on every version passed.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHONS=("$@")
if [[ ${#PYTHONS[@]} -eq 0 ]]; then
    PYTHONS=(3.11 3.12 3.13)
fi

WORK="$(mktemp -d -t rlm-studio-smoke-XXXXXX)"
FAILURES=0
PASSES=0

# shellcheck disable=SC2317  # invoked via trap
cleanup() {
    for pidfile in "$WORK"/*.pid; do
        [[ -f "$pidfile" ]] || continue
        pid="$(cat "$pidfile")"
        kill "$pid" 2>/dev/null
        wait "$pid" 2>/dev/null
    done
    rm -rf "$WORK"
}
trap cleanup EXIT

ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$1"; PASSES=$((PASSES + 1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAILURES=$((FAILURES + 1)); }
note() { printf '        %s\n' "$1"; }
head_() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

check() {  # check <description> <command...>
    local desc="$1"; shift
    if "$@" >/dev/null 2>&1; then ok "$desc"; else bad "$desc"; fi
}

free_port() {
    python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()'
}

# ---------------------------------------------------------------------------
# 1. Build the bundled wheel
# ---------------------------------------------------------------------------
if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    head_ "Building the bundled Studio UI and the wheel"
    if ! (cd frontend && npm ci --silent && npm run build:bundle --silent) >"$WORK/npm.log" 2>&1; then
        bad "frontend bundle build (see $WORK/npm.log)"; tail -5 "$WORK/npm.log"; exit 1
    fi
    ok "frontend static export built"

    # Wipe the previous payload but keep the package marker — without
    # __init__.py the wheel silently ships without the UI (see RELEASING.md).
    find src/rlmstudio/_ui -mindepth 1 -not -name __init__.py -delete
    cp -r frontend/out/. src/rlmstudio/_ui/
    check "bundle copied into src/rlmstudio/_ui/" test -f src/rlmstudio/_ui/index.html
    check "_ui package marker survived the copy" test -f src/rlmstudio/_ui/__init__.py

    # Learn reads docs from the package in an install (no repo docs/ there).
    find src/rlmstudio/_docs -mindepth 1 -not -name __init__.py -delete
    cp -r docs/. src/rlmstudio/_docs/
    check "docs copied into src/rlmstudio/_docs/" test -f src/rlmstudio/_docs/troubleshoot.yaml
    check "_docs package marker survived the copy" test -f src/rlmstudio/_docs/__init__.py

    rm -rf dist
    if ! uv build >"$WORK/build.log" 2>&1; then
        bad "uv build (see $WORK/build.log)"; tail -5 "$WORK/build.log"; exit 1
    fi
    ok "wheel + sdist built"
else
    head_ "Reusing existing dist/ (SKIP_BUILD=1)"
fi

WHEEL="$(ls dist/*.whl 2>/dev/null | head -1)"
if [[ -z "$WHEEL" ]]; then bad "no wheel in dist/"; exit 1; fi
note "wheel: $WHEEL"

# ---------------------------------------------------------------------------
# 2. Wheel contents
# ---------------------------------------------------------------------------
head_ "Wheel contents"
if uv run --no-project python - "$WHEEL" >"$WORK/wheel.log" 2>&1 <<'PY'; then
import sys, zipfile
whl = sys.argv[1]
z = zipfile.ZipFile(whl); names = z.namelist()
assert "rlmstudio/_ui/index.html" in names, "bundled UI missing from wheel"
assert "rlmstudio/_ui/__init__.py" in names, "_ui package marker missing"
assert "rlmstudio/_docs/rlm-concepts.md" in names, "bundled docs missing from wheel"
assert "rlmstudio/_docs/troubleshoot.yaml" in names, "troubleshoot source missing from wheel"
assert any(n.startswith("rlmstudio/prompts/") and n.endswith(".yaml") for n in names), "prompts missing"
legacy = [n for n in names if n.startswith("rlmkit/")]
assert not legacy, f"legacy rlmkit/ paths in wheel: {legacy[:5]}"
ep = [n for n in names if n.endswith("entry_points.txt")][0]
assert "rlm-studio = rlmstudio.cli.main:main" in z.read(ep).decode(), "wrong console script"
PY
    ok "bundled UI, prompts, entry point present; no legacy paths"
else
    bad "wheel contents"; sed -n '$p' "$WORK/wheel.log"
fi

# ---------------------------------------------------------------------------
# 3. Per-Python install + boot
# ---------------------------------------------------------------------------
for PY in "${PYTHONS[@]}"; do
    head_ "Python $PY"
    VENV="$WORK/venv-$PY"

    if ! uv venv --python "$PY" "$VENV" >"$WORK/venv-$PY.log" 2>&1; then
        bad "python $PY unavailable (uv venv failed)"; continue
    fi
    if ! uv pip install --python "$VENV/bin/python" "${WHEEL}[studio]" >"$WORK/pip-$PY.log" 2>&1; then
        bad "wheel install [studio]"; tail -3 "$WORK/pip-$PY.log"; continue
    fi
    ok "wheel installs with the [studio] extra"

    VERSION_OUT="$("$VENV/bin/rlm-studio" version 2>&1)"
    if [[ "$VERSION_OUT" == rlm-studio\ * ]]; then
        ok "rlm-studio version → $VERSION_OUT"
    else
        bad "rlm-studio version → $VERSION_OUT"
    fi

    check "public API imports" "$VENV/bin/python" -c \
        "import rlmstudio; from rlmstudio import interact; import rlmstudio.server.app"

    # Extras hygiene: a user install must not drag in the dev toolchain.
    LEAKED="$("$VENV/bin/python" - <<'PY'
import importlib.util
leaks = [m for m in ("pytest", "mypy", "ruff", "bandit", "streamlit", "plotly")
         if importlib.util.find_spec(m) is not None]
print(",".join(leaks))
PY
)"
    if [[ -z "$LEAKED" ]]; then
        ok "no dev/Streamlit packages leaked into the user install"
    else
        bad "packages leaked into [studio] install: $LEAKED"
    fi

    # Boot the one-click path with isolated state.
    PORT="$(free_port)"
    STATE="$WORK/state-$PY"
    RLM_STUDIO_DIR="$STATE" "$VENV/bin/rlm-studio" studio --no-browser --port "$PORT" \
        >"$WORK/studio-$PY.log" 2>&1 &
    echo $! >"$WORK/studio-$PY.pid"

    UP=0
    for _ in $(seq 1 60); do
        if curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then UP=1; break; fi
        sleep 1
    done

    if [[ $UP -eq 1 ]]; then
        ok "rlm-studio studio boots and serves /health"

        code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/studio/")
        [[ "$code" == "200" ]] && ok "/studio/ serves the bundled UI (200)" \
                               || bad "/studio/ returned $code"

        if curl -s "http://127.0.0.1:$PORT/studio/" | grep -q "RLM Studio"; then
            ok "bundled UI carries the RLM Studio brand"
        else
            bad "bundled UI HTML does not contain 'RLM Studio'"
        fi

        asset=$(curl -s "http://127.0.0.1:$PORT/studio/" \
                | grep -o '/studio/_next/static/[^"]*\.js' | head -1)
        if [[ -n "$asset" ]]; then
            code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT$asset")
            [[ "$code" == "200" ]] && ok "hashed _next asset serves (200)" \
                                   || bad "hashed asset returned $code"
        else
            bad "no hashed _next asset referenced by the page"
        fi

        for doc in rlm-concepts hosts-lmstudio troubleshoot; do
            code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/api/docs/$doc")
            [[ "$code" == "200" ]] && ok "Learn doc '$doc' serves from the package (200)" \
                                   || bad "Learn doc '$doc' returned $code"
        done

        # The isolated state dir proves RLM_STUDIO_DIR is honoured.
        [[ -d "$STATE" ]] && ok "RLM_STUDIO_DIR honoured (state in $STATE)" \
                          || bad "RLM_STUDIO_DIR ignored — nothing written to $STATE"

        if grep -qiE "traceback|error" "$WORK/studio-$PY.log"; then
            bad "errors in the server log (see $WORK/studio-$PY.log)"
        else
            ok "clean server log"
        fi
    else
        bad "rlm-studio studio did not come up on port $PORT"
        tail -5 "$WORK/studio-$PY.log"
    fi

    # `wait` after the kill keeps the shell from printing its own
    # "Terminated: 15" job notice over the summary.
    studio_pid="$(cat "$WORK/studio-$PY.pid")"
    kill "$studio_pid" 2>/dev/null
    wait "$studio_pid" 2>/dev/null
    rm -f "$WORK/studio-$PY.pid"
done

# ---------------------------------------------------------------------------
head_ "Summary"
printf '  %d passed, %d failed\n' "$PASSES" "$FAILURES"
if [[ $FAILURES -eq 0 ]]; then
    printf '  \033[32mAll install-path checks passed.\033[0m\n'
    printf '  Reminder: src/rlmstudio/_ui/ and _docs/ now hold build copies — they must\n'
    printf '  NOT be committed. Restore them with:\n'
    printf '    find src/rlmstudio/_ui src/rlmstudio/_docs -mindepth 1 -not -name __init__.py -delete\n\n'
    exit 0
fi
printf '  \033[31mSome checks failed — logs under %s\033[0m\n\n' "$WORK"
exit 1
