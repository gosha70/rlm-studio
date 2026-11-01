#!/usr/bin/env python3
"""Live-provider check for the litellm 1.89.7 -> 1.103.x upgrade (PR #91).

Every automated test in this repository mocks the LLM, by design, so a
green suite says nothing about whether litellm's real request path still
works after a fourteen-minor-version jump. This drives the running
server's own HTTP API -- the same endpoints the Studio UI calls -- so
nothing here re-implements provider resolution or pricing.

Usage:

    uv run python -m rlmstudio.server          # terminal 1
    uv run python live_check.py --list         # show Chat Providers, run nothing
    uv run python live_check.py                # first enabled provider
    uv run python live_check.py --provider NAME

Exits non-zero if any check fails. Spends real tokens on a real key:
two short turns, a few thousand tokens.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

DEFAULT_BASE = "http://127.0.0.1:8000"
POLL_SECONDS = 2.0
# RLM runs several model calls in a loop, so it needs far longer than Direct.
TIMEOUT_SECONDS = {"direct": 180, "rlm": 600}
DIRECT_PROMPT = "In one sentence, what is a recursive language model?"

# Long enough that the model has to inspect before answering, with the fact
# buried in the fourth paragraph so a single read of the opening will not
# produce it. A one-step answer is still a valid outcome, but this gives the
# loop something to do.
RLM_DOCUMENT = """\
Quarterly Operations Review — Northfield Division

The division closed the quarter with throughput broadly in line with the
forecast issued in the preceding review, though the composition of that
throughput shifted in ways the original model did not anticipate. Demand for
the legacy fulfilment line softened earlier than expected, while the retrofit
programme absorbed capacity that had been provisionally allocated elsewhere.

Staffing remained stable across the period. Voluntary attrition sat below the
rolling twelve-month average, and the two open supervisory roles carried over
from the previous quarter were both filled internally, which removed the
onboarding drag that had been assumed in the planning case.

Supplier performance was mixed. The primary coating vendor missed two
scheduled deliveries in the second month, which forced a partial resequencing
of the finishing schedule, though no downstream commitments were affected.
Secondary suppliers performed to contract throughout.

Equipment availability was the quarter's limiting factor. The Northfield
calibration bay recorded 47 unplanned downtime hours against a tolerance of
30, almost all of it concentrated in a single fortnight following the
controller firmware update. Root-cause analysis attributes the bulk of those
hours to a regression in the thermal interlock logic rather than to mechanical
wear.

Looking ahead, the retrofit programme is expected to release capacity from the
middle of next quarter onward, at which point the fulfilment line's utilisation
should normalise without further intervention.
"""

RLM_QUESTION = (
    "How many unplanned downtime hours did the Northfield calibration bay "
    "record, and what was the tolerance? Answer with both numbers."
)

# A short document for the Direct turn. /api/chat requires `content` or
# `file_ids` as well as `query`, so a question with no material attached is
# rejected — this is a workbench for querying content, not a chat box.
DIRECT_DOCUMENT = (
    "A recursive language model answers a question about material larger "
    "than its context window by writing code that inspects the material, "
    "reading back only what it needs, and recursing when a sub-question is "
    "itself too large to answer directly."
)


def _get(base: str, path: str) -> object:
    with urllib.request.urlopen(f"{base}{path}", timeout=30) as r:
        return json.load(r)


def _post(base: str, path: str, payload: dict[str, object]) -> object:
    req = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _run_turn(base: str, mode: str, document: str, question: str, provider_id: str | None) -> dict:
    """Submit one turn and poll its trace until it leaves `running`.

    The payload mirrors what the UI sends (frontend/src/lib/api.ts):
    ``query`` is the question and is mandatory; ``content`` carries the
    material being asked about, and the route rejects a request with
    neither ``content`` nor ``file_ids``.
    """
    payload: dict[str, object] = {"query": question, "content": document, "mode": mode}
    if provider_id:
        payload["chat_provider_id"] = provider_id

    submitted = _post(base, "/api/chat", payload)
    assert isinstance(submitted, dict)
    execution_id = submitted["execution_id"]
    print(f"  submitted {mode}: execution {execution_id}")

    deadline = time.monotonic() + TIMEOUT_SECONDS[mode]
    while time.monotonic() < deadline:
        time.sleep(POLL_SECONDS)
        try:
            trace = _get(base, f"/api/traces/{execution_id}")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                continue  # not registered yet
            raise
        assert isinstance(trace, dict)
        if trace.get("status") != "running":
            return trace
        print(f"    ... {len(trace.get('steps') or [])} steps so far", end="\r")

    raise TimeoutError(f"{mode} turn did not finish in {TIMEOUT_SECONDS[mode]}s")


def _check(results: list[tuple[bool, str]], ok: bool, label: str) -> None:
    results.append((ok, label))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def _report(mode: str, trace: dict, results: list[tuple[bool, str]]) -> None:
    result = trace.get("result") or {}
    steps = trace.get("steps") or []
    answer = (result.get("answer") or "").strip()
    tokens_in = result.get("input_tokens") or 0
    tokens_out = result.get("output_tokens") or 0
    total_cost = result.get("total_cost") or 0.0

    print(f"\n{mode.upper()} — status {trace.get('status')} (ran as mode={trace.get('mode')})")
    # Print the error first and in full. Without it a failed run looks
    # identical to a pricing regression: empty answer, zero tokens, zero cost.
    if result.get("error"):
        print(f"  ERROR: {result['error']}")
    print(f"  answer ({len(answer)} chars): {answer[:160]!r}")
    print(f"  tokens: in={tokens_in} out={tokens_out}")
    print(f"  total_cost: ${total_cost:.6f}")
    print(f"  steps: {len(steps)}")

    _check(results, result.get("success") is True, f"{mode}: run succeeded")
    _check(results, bool(answer), f"{mode}: non-empty answer")
    _check(results, (tokens_in + tokens_out) > 0, f"{mode}: non-zero tokens")
    # The whole point of the check: litellm's pricing map must still resolve
    # the model name the adapter passes. A real answer with real tokens and a
    # zero cost means the lookup silently stopped matching.
    _check(results, total_cost > 0, f"{mode}: cost is known (> 0)")

    if mode == "rlm":
        # A single-step RLM run is a valid outcome, not a regression — the
        # model may simply answer without inspecting.
        _check(results, len(steps) >= 1, "rlm: at least one step recorded")

        for s in steps:
            print(
                f"    step {s.get('index')}: {s.get('action_type'):<8}"
                f" in={s.get('input_tokens')} out={s.get('output_tokens')}"
                f" ${float(s.get('cost_usd') or 0.0):.8f}"
            )

        # Not every row calls the model, so demanding tokens on all of them
        # fails for the wrong reason: step 0 is the runtime fingerprint, and
        # `subcall` rows attribute their tokens to the rows that made the
        # request. Requiring at least one billed row keeps the check honest.
        billed = [
            s
            for s in steps
            if (int(s.get("input_tokens") or 0) + int(s.get("output_tokens") or 0)) > 0
        ]
        _check(results, len(billed) >= 1, "rlm: at least one step carries tokens")

        # The real per-step invariant. Token accounting IS populated per step,
        # and the rows that carry tokens must reconcile exactly with the run
        # total — that is what catches per-step accounting drifting away from
        # the client, which is what the cost check was reaching for.
        step_in = sum(int(s.get("input_tokens") or 0) for s in steps)
        step_out = sum(int(s.get("output_tokens") or 0) for s in steps)
        print(f"  per-step token sum: in={step_in} out={step_out}")
        _check(
            results,
            step_in == int(tokens_in) and step_out == int(tokens_out),
            f"rlm: per-step tokens sum to the run total "
            f"(steps {step_in}/{step_out} vs run {tokens_in}/{tokens_out})",
        )

        # Per-step cost is NOT asserted: TraceStep is built without ever
        # setting `cost_usd` (server/routes/traces.py, both construction
        # sites), so it is always 0.0 regardless of the provider or the
        # litellm version. Reported for visibility, not checked — see the
        # open issue. Run-level cost is asserted above, where it is real.
        step_cost = sum(float(s.get("cost_usd") or 0.0) for s in steps)
        if step_cost == 0.0 and float(total_cost) > 0:
            print(
                f"  note: per-step cost is $0.00 on every row while the run "
                f"total is ${total_cost:.6f} — known gap, not a regression."
            )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=DEFAULT_BASE, help=f"server URL (default {DEFAULT_BASE})")
    ap.add_argument(
        "--provider", default=None, help="alias for --direct-provider (kept for convenience)"
    )
    ap.add_argument(
        "--direct-provider", default=None, help="Chat Provider (mode `direct`) for the Direct turn"
    )
    ap.add_argument(
        "--rlm-provider", default=None, help="Chat Provider (mode `rlm`) for the RLM turn"
    )
    ap.add_argument("--list", action="store_true", help="list Chat Providers and exit")
    args = ap.parse_args(argv)

    try:
        _get(args.base, "/health")
    except Exception as exc:
        print(f"Cannot reach {args.base}: {exc}", file=sys.stderr)
        print("Start it with: uv run python -m rlmstudio.server", file=sys.stderr)
        return 2

    providers = _get(args.base, "/api/chat-providers")
    if not isinstance(providers, list) or not providers:
        print("No Chat Providers configured. Add one in Settings first.", file=sys.stderr)
        return 2

    if args.list:
        print(f"{'name':28} {'mode':14} {'enabled':8} id")
        for p in providers:
            print(
                f"{str(p.get('name')):28} {str(p.get('execution_mode')):14}"
                f" {str(p.get('enabled', True)):8} {p.get('id')}"
            )
        return 0

    # A Chat Provider's own execution_mode wins over the request's `mode`
    # (routes/chat.py: `mode = cp.execution_mode`), so one provider cannot
    # drive both turns — pointing an rlm provider at the Direct turn silently
    # runs RLM twice. Pick one provider per mode instead.
    def _pick(wanted_mode: str, requested: str | None) -> dict | None:
        if requested:
            p = next((q for q in providers if requested in (q.get("name"), q.get("id"))), None)
            if p is None:
                print(f"No Chat Provider named {requested!r}.", file=sys.stderr)
            elif p.get("execution_mode") != wanted_mode:
                print(
                    f"Chat Provider {requested!r} is mode "
                    f"{p.get('execution_mode')!r}, not {wanted_mode!r}; its own "
                    "mode would win. Pick one in that mode.",
                    file=sys.stderr,
                )
                return None
            return p
        return next(
            (
                q
                for q in providers
                if q.get("execution_mode") == wanted_mode and q.get("enabled", True)
            ),
            None,
        )

    direct_cp = _pick("direct", args.direct_provider or args.provider)
    rlm_cp = _pick("rlm", args.rlm_provider)
    if direct_cp is None or rlm_cp is None:
        print(
            "\nNeed one Chat Provider in `direct` mode and one in `rlm` mode.\n"
            "Run with --list to see them, then pass --direct-provider / --rlm-provider.",
            file=sys.stderr,
        )
        return 2

    print(f"Direct via: {direct_cp.get('name')} ({direct_cp.get('id')})")
    print(f"RLM via:    {rlm_cp.get('name')} ({rlm_cp.get('id')})")
    print("Two turns against a real provider. This spends tokens on your key.\n")

    results: list[tuple[bool, str]] = []
    turns = (
        ("direct", DIRECT_DOCUMENT, DIRECT_PROMPT, direct_cp),
        ("rlm", RLM_DOCUMENT, RLM_QUESTION, rlm_cp),
    )
    for mode, document, question, cp in turns:
        try:
            trace = _run_turn(args.base, mode, document, question, cp.get("id"))
        except urllib.error.HTTPError as exc:
            # Show the body: a bare "422 Unprocessable Content" hides the
            # field-level detail that says what the payload got wrong.
            body = exc.read().decode(errors="replace")[:500]
            print(f"\n{mode.upper()} — HTTP {exc.code}: {body}")
            results.append((False, f"{mode}: turn completed"))
            continue
        except Exception as exc:
            print(f"\n{mode.upper()} — raised {type(exc).__name__}: {exc}")
            results.append((False, f"{mode}: turn completed"))
            continue
        _report(mode, trace, results)

    failed = [label for ok, label in results if not ok]
    print(f"\n{'=' * 60}")
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("\nFailed:")
        for label in failed:
            print(f"  - {label}")
        # Only point at pricing when the run actually reached the model.
        # A failed run also shows zero cost, and blaming the pricing map for
        # an auth error or a bad payload sends the reader to the wrong file.
        reached_model = any("non-zero tokens" in label for ok, label in results if ok)
        if reached_model and any("cost is known" in f for f in failed):
            print(
                "\nA real answer with real tokens but zero cost means litellm's\n"
                "pricing map no longer resolves the model name the adapter\n"
                "passes. Look at how the provider prefix is stripped before the\n"
                "lookup in infrastructure/llm/litellm_adapter.py — not at the lock."
            )
        elif not reached_model:
            print(
                "\nZero tokens on every turn means no request reached the model,\n"
                "so this says nothing about the litellm upgrade. Read the ERROR\n"
                "line above: an authentication error means the provider's key is\n"
                "invalid or expired, which is configuration, not code."
            )
        return 1
    print("\nAll checks passed. PR #91 has nothing left blocking it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
