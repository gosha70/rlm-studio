"""Real ``rlms`` runs against a scripted OpenAI-compatible stub.

Covers specs/interop-official-rlm T3.3: AC-1 (an ``rlm_official`` run produces
``inspect`` steps and exactly one ``final`` step through Studio's normaliser)
and AC-3 (an execution that never returns is stopped by Studio's wall-clock
budget and classified ``timeout``).

Needs the ``interop`` extra (``rlm`` importable); skipped otherwise, and
gated behind ``--runslow`` like the other heavyweight tests::

    uv run pytest tests/test_rlms_engine_real.py --runslow
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

pytest.importorskip("rlm")

from rlmstudio.application.dto import RunConfigDTO  # noqa: E402
from rlmstudio.application.sandbox_vars import (  # noqa: E402
    RESULT_KEY_ENGINE_VERSION,
    TRACE_KEY_CONTENT,
    TRACE_KEY_ROLE,
)
from rlmstudio.application.services.outcome_classifier import (  # noqa: E402
    OutcomeCategory,
    classify_execution_outcome,
)
from rlmstudio.application.use_cases.run_rlm_official import RunRLMOfficialUseCase  # noqa: E402
from rlmstudio.infrastructure.engines.rlms_adapter import RlmsEngineAdapter  # noqa: E402
from rlmstudio.server.routes._helpers import _canonical_action_type  # noqa: E402

pytestmark = pytest.mark.slow

_ANSWER = "The document has 1234 characters."
_STUB_MODEL = "stub-model"
_PROMPT_TOKENS = 100
_COMPLETION_TOKENS = 20


def _chat_completion(content: str) -> dict[str, Any]:
    """The subset of an OpenAI chat completion the ``rlms`` client reads."""
    return {
        "id": "chatcmpl-stub",
        "object": "chat.completion",
        "created": 0,
        "model": _STUB_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": _PROMPT_TOKENS,
            "completion_tokens": _COMPLETION_TOKENS,
            "total_tokens": _PROMPT_TOKENS + _COMPLETION_TOKENS,
        },
    }


class _ScriptedOpenAIStub:
    """Minimal OpenAI-compatible server on a free localhost port.

    Answers each ``POST …/chat/completions`` with the next scripted reply;
    the last reply repeats if the engine keeps asking.
    """

    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.requests: list[dict[str, Any]] = []
        stub = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 — http.server API
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                if not self.path.endswith("/chat/completions"):
                    self.send_response(404)
                    self.end_headers()
                    return
                stub.requests.append(body)
                index = min(len(stub.requests), len(stub.replies)) - 1
                payload = json.dumps(_chat_completion(stub.replies[index])).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args: Any) -> None:  # silence the request log
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        port = self._server.server_address[1]
        return f"http://127.0.0.1:{port}/v1"

    def __enter__(self) -> _ScriptedOpenAIStub:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()


def _adapter(base_url: str) -> RlmsEngineAdapter:
    # ``lmstudio`` = an OpenAI-compatible local server; the stub is exactly that.
    return RlmsEngineAdapter(
        backend="lmstudio",
        model=_STUB_MODEL,
        base_url=base_url,
        sandbox_type="restricted",  # → rlms 'local' environment
        cost_fn=lambda input_tokens, output_tokens: 0.0,
    )


def test_real_engine_run_produces_inspect_steps_and_one_final() -> None:
    """AC-1: the trace normalises to ≥1 ``inspect`` and exactly 1 ``final`` step."""
    replies = [
        "Let me measure the context first.\n```repl\nprint(len(context))\n```",
        f'```repl\nanswer["content"] = "{_ANSWER}"\nanswer["ready"] = True\n```',
    ]
    with _ScriptedOpenAIStub(replies) as stub:
        result = RunRLMOfficialUseCase(_adapter(stub.base_url)).execute(
            "x" * 1234,
            "How long is the document?",
            RunConfigDTO(max_steps=5, max_time_seconds=60.0),
        )

    assert result.success, result.error
    assert result.answer == _ANSWER
    assert result.steps == 2
    assert len(stub.requests) == 2
    assert result.metadata[RESULT_KEY_ENGINE_VERSION]

    # Usage is summed over both root calls.
    assert result.input_tokens == 2 * _PROMPT_TOKENS
    assert result.output_tokens == 2 * _COMPLETION_TOKENS

    # The REPL really ran: the printed length shows up as an execution entry.
    executions = [e for e in result.trace if e[TRACE_KEY_ROLE] == "execution"]
    assert any("1234" in e[TRACE_KEY_CONTENT] for e in executions)

    actions = [
        _canonical_action_type(e[TRACE_KEY_ROLE], is_last=i == len(result.trace) - 1, success=True)
        for i, e in enumerate(result.trace)
    ]
    assert actions.count("inspect") >= 1
    assert actions.count("final") == 1
    assert result.trace[-1][TRACE_KEY_CONTENT] == _ANSWER


def test_wall_clock_budget_stops_a_stuck_repl_execution() -> None:
    """AC-3: an execution that never returns within budget is classified ``timeout``.

    The engine's own ``max_timeout`` is checked *between* iterations, so an
    execution stuck inside one iteration is only caught by Studio's guard.
    ``time.sleep`` stands in for ``while True: pass`` deliberately: the engine
    runs the code in-process, and the abandoned daemon thread would otherwise
    hold the GIL in a tight loop and slow every other test in the worker.
    The mechanism under test is identical.
    """
    replies = ["```repl\nimport time\ntime.sleep(30)\n```"]
    with _ScriptedOpenAIStub(replies) as stub:
        result = RunRLMOfficialUseCase(_adapter(stub.base_url)).execute(
            "doc",
            "q",
            RunConfigDTO(max_steps=3, max_time_seconds=2.0),
        )

    assert not result.success
    assert result.elapsed_time < 10.0
    outcome = classify_execution_outcome(result.success, result.error, result.answer)
    assert outcome.category is OutcomeCategory.TIMEOUT
