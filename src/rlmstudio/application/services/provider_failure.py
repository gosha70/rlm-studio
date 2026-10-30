"""Classify an LLM provider failure once, so every surface says the same thing.

Two code paths reach a provider and can fail: the run path, through
``LiteLLMAdapter._translate_exception``, and the connection probe, through
``provider_tester.test_provider``. They do not share a call stack — the probe
calls ``litellm.completion`` itself with its own timeout contract — so without
a shared classifier each surface invents its own wording, and both historically
showed the raw upstream exception.

This module is the single classification point. It lives in the application
layer rather than beside the adapter because ``provider_tester`` is also
application-level, and the dependency rule forbids application importing from
infrastructure. Nothing here imports litellm: classification works from the
exception's class name and message text, which is also what lets the probe
classify a failure it caught from a bare ``litellm.completion`` call.

See issue #95. Model-list validation and the background-probe default are
deliberately out of scope here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "ProviderFailureKind",
    "ProviderFailure",
    "ProviderFailureError",
    "classify_provider_failure",
    "describe_provider_failure",
    "failure_kind_of",
    "failure_kind_value",
    "redact_secrets",
    "SECRET_PLACEHOLDER",
]

# nosec B105 — this is what a redacted secret is replaced *with*, not a
# credential. Bandit's hardcoded-password heuristic matches the name.
SECRET_PLACEHOLDER = "<redacted>"  # nosec B105

# Upstream auth errors sometimes echo the key that was rejected, and
# httpx-layer errors can echo request headers including Authorization. Both
# the run path and the probe now surface auth failures to users, so the
# patterns live here with the classifier rather than in one caller.
_API_KEY_PATTERNS: tuple[re.Pattern[str], ...] = (
    # OpenAI-style sk-..., sk-proj-..., etc.
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    # Anthropic-style
    re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
    # Generic Bearer token in an HTTP header echoed back
    re.compile(r"[Bb]earer\s+[A-Za-z0-9_.~+/=-]{20,}"),
    # Hugging Face tokens
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    # Google-style
    re.compile(r"AIza[A-Za-z0-9_-]{20,}"),
)


def redact_secrets(text: str) -> str:
    """Replace anything matching a known API-key shape with a placeholder.

    Never returns the original string when a pattern matches. Callers that
    show upstream exception text to a user must pass it through here first.
    """
    for pattern in _API_KEY_PATTERNS:
        text = pattern.sub(SECRET_PLACEHOLDER, text)
    return text


class ProviderFailureKind(str, Enum):
    """What went wrong when talking to a provider.

    ``str`` mixin so the value serialises directly into an API payload and
    the UI can switch on it to link at the right remedy.
    """

    AUTH = "auth"
    MODEL_UNAVAILABLE = "model_unavailable"
    CONNECTION = "connection"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    UNKNOWN = "unknown"


# litellm raises provider-agnostic exception classes. Matching on the class
# name rather than importing litellm keeps this module dependency-free and
# layer-clean; the names are stable across the 1.x line and are checked
# before message keywords because they are unambiguous.
_CLASS_NAME_KINDS: dict[str, ProviderFailureKind] = {
    "AuthenticationError": ProviderFailureKind.AUTH,
    "PermissionDeniedError": ProviderFailureKind.AUTH,
    "NotFoundError": ProviderFailureKind.MODEL_UNAVAILABLE,
    "APIConnectionError": ProviderFailureKind.CONNECTION,
    "APIConnectionTimeoutError": ProviderFailureKind.TIMEOUT,
    "Timeout": ProviderFailureKind.TIMEOUT,
    "RateLimitError": ProviderFailureKind.RATE_LIMIT,
}

# Fallbacks for providers or httpx layers that surface a generic exception
# class with a descriptive message. Ordered: the first kind whose keywords
# match wins, so the more specific phrases come first.
#
# Bare status numbers ("401", "429") are deliberately NOT here. They match as
# substrings of ordinary figures in unrelated messages — "140123 tokens"
# contains "401", "max_tokens ... 4290" contains "429" — which classified a
# context-window overflow as an auth failure and told a user with an oversized
# document to rotate a working API key. A confidently wrong remedy is worse
# than the raw detail this replaced. Real auth and rate-limit failures are
# caught by class name above; a provider that reports only a bare status code
# in free text falls through to UNKNOWN and keeps its detail, which is the
# honest outcome.
_KEYWORD_KINDS: tuple[tuple[ProviderFailureKind, tuple[str, ...]], ...] = (
    (
        ProviderFailureKind.AUTH,
        (
            "invalid x-api-key",
            "invalid api key",
            "authentication_error",
            "unauthorized",
            "invalid authentication",
        ),
    ),
    (
        ProviderFailureKind.MODEL_UNAVAILABLE,
        ("not_found_error", "model not found", "does not exist", "unknown model"),
    ),
    (
        ProviderFailureKind.RATE_LIMIT,
        ("rate limit", "rate_limit", "too many requests", "quota"),
    ),
)


@dataclass(frozen=True)
class ProviderFailure:
    """A classified failure, with text a user can act on.

    ``cause`` says what went wrong in the user's terms; ``fix`` says what to
    do about it. Neither contains the upstream exception — callers append
    their own sanitised detail, because the sanitisation rule (redact API
    keys, cap length) lives with the caller that already owns it.
    """

    kind: ProviderFailureKind
    cause: str
    fix: str


def classify_provider_failure(exc: BaseException) -> ProviderFailureKind:
    """Return the kind of provider failure *exc* represents.

    Class name first (unambiguous), then message keywords (for providers
    that raise a generic class with a descriptive body). Falls back to
    ``UNKNOWN`` rather than guessing, so an unrecognised failure keeps its
    raw detail instead of being mislabelled.
    """
    for cls in type(exc).__mro__:
        kind = _CLASS_NAME_KINDS.get(cls.__name__)
        if kind is not None:
            return kind

    message = str(exc).lower()
    for kind, keywords in _KEYWORD_KINDS:
        if any(kw in message for kw in keywords):
            return kind

    return ProviderFailureKind.UNKNOWN


def describe_provider_failure(
    kind: ProviderFailureKind,
    *,
    provider_name: str | None = None,
    model: str | None = None,
    api_base: str | None = None,
) -> ProviderFailure:
    """Return the user-facing cause and fix for *kind*.

    All arguments are optional so the probe, which knows the provider and
    model, and the run path, which may know only the model, can both call
    it without inventing placeholder values.
    """
    where = f" for {provider_name}" if provider_name else ""
    named_model = f" '{model}'" if model else ""

    if kind is ProviderFailureKind.AUTH:
        return ProviderFailure(
            kind=kind,
            cause=f"The API key{where} was rejected.",
            fix=(
                "Check the key has not been revoked or regenerated, then update "
                "it in Settings → Providers."
            ),
        )

    if kind is ProviderFailureKind.MODEL_UNAVAILABLE:
        # Deliberately not "retired". A 404 on a model also covers a typo, a
        # rename, and a model the account has no entitlement for; asserting
        # retirement would send the user to the wrong remedy.
        return ProviderFailure(
            kind=kind,
            cause=(
                f"The provider does not offer model{named_model} to this account. "
                "It may have been retired or renamed, the name may be misspelled, "
                "or the account may not have access to it."
            ),
            fix=(
                "Pick a model from the provider's current list in "
                "Settings → Providers, which is fetched from your account."
            ),
        )

    if kind is ProviderFailureKind.CONNECTION:
        location = f" at '{api_base}'" if api_base else ""
        return ProviderFailure(
            kind=kind,
            cause=f"Could not reach the provider{location}.",
            fix=(
                "Make sure the server is running and reachable."
                if api_base
                else "Check network connectivity and the provider's endpoint."
            ),
        )

    if kind is ProviderFailureKind.TIMEOUT:
        return ProviderFailure(
            kind=kind,
            cause="The provider did not respond in time.",
            fix=(
                "Raise the timeout in the runtime settings, or try a smaller "
                "request or a faster model."
            ),
        )

    if kind is ProviderFailureKind.RATE_LIMIT:
        return ProviderFailure(
            kind=kind,
            cause=f"The provider is rate-limiting or out of quota{where}.",
            fix="Wait and retry, or check the account's usage limits with the provider.",
        )

    return ProviderFailure(
        kind=ProviderFailureKind.UNKNOWN,
        cause="The provider call failed.",
        fix="See the detail below; if it is not clear, check Settings → Providers.",
    )


class ProviderFailureError(RuntimeError):
    """A classified provider failure, raised by the run path.

    Subclasses ``RuntimeError`` deliberately: the use cases and routes
    already catch ``RuntimeError`` and read ``str(exc)``, and all of that
    keeps working unchanged. The added attributes let the server attach a
    stable ``failure_kind`` to the run payload, so the UI can link to the
    right remedy without matching on wording — which is the trap the enum
    exists to avoid.
    """

    def __init__(self, message: str, failure: ProviderFailure) -> None:
        super().__init__(message)
        self.kind = failure.kind
        self.cause_text = failure.cause
        self.fix_text = failure.fix


def failure_kind_of(exc: BaseException) -> ProviderFailureKind | None:
    """Return the carried failure kind, or ``None`` for anything else.

    Lets a generic ``except Exception`` handler attach the classification
    without having to know about ``ProviderFailureError`` or re-classify an
    exception that has already travelled through the adapter.
    """
    kind = getattr(exc, "kind", None)
    return kind if isinstance(kind, ProviderFailureKind) else None


def failure_kind_value(exc: BaseException) -> str | None:
    """Return the carried failure kind as its string value, or ``None``.

    Convenience for the use cases, which catch provider exceptions
    themselves and build a result DTO rather than letting the route's
    handler see them. Returning the plain string keeps the DTO and the API
    payload free of an enum import.
    """
    kind = failure_kind_of(exc)
    return kind.value if kind is not None else None
