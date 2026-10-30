"""Tests for the shared provider-failure classifier (issue #95).

Two paths reach a provider and can fail — the run path through
``LiteLLMAdapter._translate_exception`` and the connection probe through
``provider_tester.test_provider`` — and they do not share a call stack. These
tests pin that both classify through the same function, that the wording is
actionable, that a rejected key is never echoed, and that the pre-existing
timeout contract three use cases depend on is unchanged.
"""

from __future__ import annotations

import pytest

from rlmstudio.application.services.provider_failure import (
    SECRET_PLACEHOLDER,
    ProviderFailureKind,
    classify_provider_failure,
    describe_provider_failure,
    redact_secrets,
)

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_classifies_the_real_litellm_exception_classes() -> None:
    """The two failures actually hit while verifying the litellm upgrade.

    Classification keys on the exception class name, so this pins the
    mapping against the classes litellm really raises rather than against
    hand-written stand-ins.
    """
    litellm = pytest.importorskip("litellm")

    auth = litellm.AuthenticationError(
        'AnthropicException - {"type":"authentication_error","message":"invalid x-api-key"}',
        llm_provider="anthropic",
        model="claude-sonnet-4-5-20250929",
    )
    not_found = litellm.NotFoundError(
        'AnthropicException - {"type":"not_found_error","message":"model: claude-sonnet-4"}',
        llm_provider="anthropic",
        model="claude-sonnet-4",
    )

    assert classify_provider_failure(auth) is ProviderFailureKind.AUTH
    assert classify_provider_failure(not_found) is ProviderFailureKind.MODEL_UNAVAILABLE


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("invalid x-api-key", ProviderFailureKind.AUTH),
        ("authentication_error from upstream", ProviderFailureKind.AUTH),
        ('{"type":"not_found_error","message":"model: x"}', ProviderFailureKind.MODEL_UNAVAILABLE),
        ("the model does not exist", ProviderFailureKind.MODEL_UNAVAILABLE),
        ("rate limit exceeded", ProviderFailureKind.RATE_LIMIT),
    ],
)
def test_falls_back_to_message_keywords(message: str, expected: ProviderFailureKind) -> None:
    """Providers that raise a generic class with a descriptive body still
    classify, via keywords, so the probe is not blind to them."""
    assert classify_provider_failure(Exception(message)) is expected


def test_unrecognised_failure_stays_unknown() -> None:
    """An unfamiliar failure must not be mislabelled — UNKNOWN keeps the raw
    detail visible instead of attaching a confident, wrong remedy."""
    assert classify_provider_failure(Exception("something odd happened")) is (
        ProviderFailureKind.UNKNOWN
    )


# ---------------------------------------------------------------------------
# Wording
# ---------------------------------------------------------------------------


def test_model_message_does_not_assert_retirement() -> None:
    """A 404 on a model is ambiguous: retired, renamed, misspelled, or not
    entitled. The message must not claim retirement as the cause."""
    failure = describe_provider_failure(
        ProviderFailureKind.MODEL_UNAVAILABLE, model="claude-sonnet-4-20250514"
    )

    assert "claude-sonnet-4-20250514" in failure.cause
    # It may mention retirement as one possibility, but never as the fact.
    assert "may have been retired" in failure.cause
    assert "has been retired" not in failure.cause
    assert "is retired" not in failure.cause
    # And it must point at the account's own list, not a guess.
    assert "Settings" in failure.fix


def test_every_kind_has_a_cause_and_a_fix() -> None:
    """No kind may fall through to empty text — a classified failure the UI
    cannot explain is worse than an unclassified one."""
    for kind in ProviderFailureKind:
        failure = describe_provider_failure(kind)
        assert failure.cause.strip(), f"{kind} has no cause"
        assert failure.fix.strip(), f"{kind} has no fix"
        assert failure.kind is kind


def test_auth_message_names_the_provider_when_known() -> None:
    with_name = describe_provider_failure(ProviderFailureKind.AUTH, provider_name="Anthropic")
    without = describe_provider_failure(ProviderFailureKind.AUTH)

    assert "Anthropic" in with_name.cause
    # The probe knows the provider; the run path may not. Neither may emit a
    # placeholder like "None".
    assert "None" not in without.cause


# ---------------------------------------------------------------------------
# Redaction — the run path now shows upstream text, so this is load-bearing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "secret",
    [
        "sk-" + "A" * 30,
        "sk-ant-" + "B" * 40,
        "Bearer " + "C" * 30,
        "hf_" + "D" * 25,
        "AIza" + "E" * 30,
    ],
)
def test_redacts_every_known_key_shape(secret: str) -> None:
    out = redact_secrets(f"rejected credential {secret} upstream")

    assert secret not in out
    assert SECRET_PLACEHOLDER in out


def test_redaction_leaves_ordinary_text_alone() -> None:
    text = "model: claude-sonnet-4-20250514 not found"
    assert redact_secrets(text) == text


# ---------------------------------------------------------------------------
# Adapter integration
# ---------------------------------------------------------------------------


def test_adapter_message_is_actionable_and_redacted() -> None:
    """The run path must surface cause and fix, and never the rejected key.

    litellm auth errors do echo the key that was rejected, so this is the
    case that would leak it into the chat error panel.
    """
    litellm = pytest.importorskip("litellm")
    from rlmstudio.infrastructure.llm.litellm_adapter import LiteLLMAdapter

    secret = "sk-ant-" + "A" * 40
    adapter = LiteLLMAdapter(model="claude-sonnet-4-20250514")
    err = adapter._translate_exception(
        litellm.AuthenticationError(
            f"AnthropicException - invalid x-api-key: {secret}",
            llm_provider="anthropic",
            model="claude-sonnet-4-20250514",
        ),
        "LiteLLM completion failed",
    )

    text = str(err)
    assert secret not in text
    assert SECRET_PLACEHOLDER in text
    assert "API key" in text
    assert "Settings" in text


def test_adapter_preserves_the_timeout_prefix() -> None:
    """``run_direct``, ``run_rag`` and ``run_rlm`` all detect timeouts with
    ``str(exc).startswith(TIMEOUT_ERROR_PREFIX)``. Classifying timeouts
    through the new path would silently break all three, so the timeout
    branch must keep its exact message shape."""
    from rlmstudio.infrastructure.llm.litellm_adapter import (
        TIMEOUT_ERROR_PREFIX,
        LiteLLMAdapter,
    )

    adapter = LiteLLMAdapter(model="gpt-4o-mini")
    err = adapter._translate_exception(Exception("Request timed out"), "LiteLLM failed")

    assert str(err).startswith(TIMEOUT_ERROR_PREFIX)


def test_adapter_unknown_failure_keeps_the_default_prefix() -> None:
    """An unclassified failure keeps the caller's prefix and its detail,
    rather than being dressed up with a remedy that may not apply."""
    from rlmstudio.infrastructure.llm.litellm_adapter import LiteLLMAdapter

    adapter = LiteLLMAdapter(model="gpt-4o-mini")
    err = adapter._translate_exception(Exception("something odd"), "LiteLLM completion failed")

    assert str(err).startswith("LiteLLM completion failed:")
    assert "something odd" in str(err)


# ---------------------------------------------------------------------------
# The enum must survive the trip to the run payload (#95 scope 3)
# ---------------------------------------------------------------------------


def test_provider_failure_error_is_a_runtime_error() -> None:
    """Every existing handler catches RuntimeError and reads str(exc).

    If the carrying exception were not a RuntimeError subclass, each of those
    handlers would stop catching provider failures — a far worse regression
    than the problem being fixed.
    """
    from rlmstudio.application.services.provider_failure import ProviderFailureError

    failure = describe_provider_failure(ProviderFailureKind.AUTH)
    exc = ProviderFailureError("boom", failure)

    assert isinstance(exc, RuntimeError)
    assert str(exc) == "boom"
    assert exc.kind is ProviderFailureKind.AUTH


def test_failure_kind_of_reads_the_carried_kind() -> None:
    """Lets a generic `except Exception` attach the classification without
    knowing about the exception type or re-classifying."""
    from rlmstudio.application.services.provider_failure import (
        ProviderFailureError,
        failure_kind_of,
    )

    carried = ProviderFailureError(
        "x", describe_provider_failure(ProviderFailureKind.MODEL_UNAVAILABLE)
    )

    assert failure_kind_of(carried) is ProviderFailureKind.MODEL_UNAVAILABLE
    # Anything else must stay None rather than being guessed at.
    assert failure_kind_of(ValueError("boom")) is None
    assert failure_kind_of(RuntimeError("boom")) is None


def test_adapter_raises_the_carrying_type_for_a_classified_failure() -> None:
    """The run path must carry the kind, not just the wording — the UI keys
    its remedy link off the enum."""
    litellm = pytest.importorskip("litellm")

    from rlmstudio.application.services.provider_failure import (
        ProviderFailureError,
        failure_kind_of,
    )
    from rlmstudio.infrastructure.llm.litellm_adapter import LiteLLMAdapter

    adapter = LiteLLMAdapter(model="claude-sonnet-4-20250514")
    err = adapter._translate_exception(
        litellm.NotFoundError(
            'AnthropicException - {"type":"not_found_error","message":"model: x"}',
            llm_provider="anthropic",
            model="claude-sonnet-4-20250514",
        ),
        "LiteLLM completion failed",
    )

    assert isinstance(err, ProviderFailureError)
    assert failure_kind_of(err) is ProviderFailureKind.MODEL_UNAVAILABLE


def test_trace_result_carries_failure_kind() -> None:
    """The field must serialise as the enum's string value, and stay null
    for anything that is not a classified provider failure."""
    from rlmstudio.server.models import TraceResult

    classified = TraceResult(answer="", success=False, error="x", failure_kind="auth")
    assert classified.model_dump()["failure_kind"] == "auth"

    plain = TraceResult(answer="hi", success=True)
    assert plain.model_dump()["failure_kind"] is None


# ---------------------------------------------------------------------------
# End to end: a provider failure must carry its kind all the way to the trace
# ---------------------------------------------------------------------------


def test_failure_kind_reaches_the_trace_through_a_use_case() -> None:
    """The gap the unit tests missed.

    The use cases catch provider exceptions themselves and return a result
    DTO, so the route's `except` block never sees them. Classifying only in
    that handler left `failure_kind` null for exactly the cases #95 is about
    — a stale key and an unavailable model. This drives the real endpoint and
    reads the real trace.
    """
    litellm = pytest.importorskip("litellm")
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    from rlmstudio.server.app import app
    from rlmstudio.server.dependencies import reset_state

    def _auth_error(*_args: object, **_kwargs: object) -> None:
        raise litellm.AuthenticationError(
            "AnthropicException - invalid x-api-key",
            llm_provider="anthropic",
            model="claude-sonnet-4-5",
        )

    reset_state()
    client = TestClient(app)

    with patch("litellm.completion", side_effect=_auth_error):
        submitted = client.post(
            "/api/chat",
            json={
                "query": "What does this say?",
                "content": "A short document.",
                "mode": "direct",
            },
        )
        assert submitted.status_code == 202, submitted.text
        execution_id = submitted.json()["execution_id"]

        trace = client.get(f"/api/traces/{execution_id}")

    assert trace.status_code == 200, trace.text
    result = trace.json()["result"]

    assert result["success"] is False
    # The wording reaches the user ...
    assert "API key" in (result["error"] or "")
    # ... and so does the classification the UI keys its remedy link off.
    assert result["failure_kind"] == "auth", (
        f"failure_kind was {result['failure_kind']!r}; the classification did "
        f"not survive the use case"
    )


def test_successful_run_has_no_failure_kind() -> None:
    """A success must leave the field null rather than reporting 'unknown'."""
    from rlmstudio.application.dto import RunResultDTO

    assert RunResultDTO(answer="hi", mode_used="direct", success=True).failure_kind is None


# ---------------------------------------------------------------------------
# Misclassification: bare status numbers matched ordinary figures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        # "140123" contains "401" — this classified as AUTH and told a user
        # with an oversized document to rotate a working API key.
        "This model's maximum context length is 200000 tokens, however you requested 140123 tokens",
        "max_tokens is too large: 4290",  # contains "429"
        "litellm.BadRequestError: max_tokens must be <= 8192, got 42900",
        "Request contains 401234 characters, which exceeds the limit",
    ],
)
def test_ordinary_numbers_are_not_mistaken_for_status_codes(message: str) -> None:
    """A figure that happens to contain 401 or 429 must not be classified.

    UNKNOWN is the right answer here: the message keeps its own detail, which
    for a context overflow is the thing that actually helps. A confidently
    wrong remedy is worse than the raw text this feature replaced.
    """
    assert classify_provider_failure(Exception(message)) is ProviderFailureKind.UNKNOWN


def test_real_auth_and_rate_limit_still_classify_by_class_name() -> None:
    """Dropping the bare numbers must not cost the real cases, which arrive
    as typed exceptions rather than as free text."""
    litellm = pytest.importorskip("litellm")

    auth = litellm.AuthenticationError("nope", llm_provider="openai", model="gpt-4o")
    rate = litellm.RateLimitError("slow down", llm_provider="openai", model="gpt-4o")

    assert classify_provider_failure(auth) is ProviderFailureKind.AUTH
    assert classify_provider_failure(rate) is ProviderFailureKind.RATE_LIMIT


def test_context_overflow_keeps_its_own_detail_through_the_adapter() -> None:
    """The user-visible outcome for an overflow: the real reason, not a
    remedy for a problem they do not have."""
    litellm = pytest.importorskip("litellm")
    from rlmstudio.infrastructure.llm.litellm_adapter import LiteLLMAdapter

    adapter = LiteLLMAdapter(model="gpt-4o-mini")
    err = adapter._translate_exception(
        litellm.ContextWindowExceededError(
            "This model's maximum context length is 128000 tokens, "
            "however you requested 140123 tokens",
            model="gpt-4o-mini",
            llm_provider="openai",
        ),
        "LiteLLM completion failed",
    )

    text = str(err)
    assert "140123 tokens" in text, "the actionable detail was dropped"
    assert "API key" not in text, "an overflow was reported as an auth failure"
