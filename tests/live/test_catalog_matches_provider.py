"""Detect retired models before a release, by asking the provider.

Everything else guarding model identifiers in this repository compares the
catalog against a *hardcoded* set of identifiers already known to be retired.
That catches a regression to a model we have previously been bitten by. It
cannot catch the next retirement: the day Anthropic retires
``claude-sonnet-4-6``, every one of those tests still passes and the first
person to learn is a user whose query fails.

Only the provider knows. These tests ask it.

They are marked ``live`` and skip without a key, so the normal suite and CI
are unaffected — CI holds no provider credentials. The release playbook runs
them before cutting (``doc_internal/release/steps/01-pre-cut-gates.md``),
which moves discovery from a user's first query to the maintainer's gate.

    uv run pytest tests/live -m live          # with ANTHROPIC_API_KEY set

Cost: list endpoints only. No completions, no tokens.
"""

from __future__ import annotations

import os

import pytest

from rlmstudio.domain.model_catalog import ANTHROPIC_MODELS, OPENAI_MODELS

pytestmark = pytest.mark.live


def _anthropic_model_names(api_key: str) -> set[str]:
    """Model ids this account can actually use, from Anthropic's list API."""
    import urllib.request

    request = urllib.request.Request(
        "https://api.anthropic.com/v1/models?limit=1000",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        import json

        payload = json.load(response)
    return {entry["id"] for entry in payload.get("data", [])}


def _openai_model_names(api_key: str) -> set[str]:
    import json
    import urllib.request

    request = urllib.request.Request(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    return {entry["id"] for entry in payload.get("data", [])}


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="needs ANTHROPIC_API_KEY; run before cutting a release",
)
def test_every_catalog_anthropic_model_is_still_offered() -> None:
    """A catalog entry the account cannot use is a retirement we missed.

    This is the check that would have caught `claude-sonnet-4-20250514`
    before it reached a user, and the one that will catch whatever replaces
    today's entries.
    """
    available = _anthropic_model_names(os.environ["ANTHROPIC_API_KEY"])
    catalogued = {m.name for m in ANTHROPIC_MODELS}

    missing = catalogued - available
    assert not missing, (
        f"the catalog offers models this account cannot use: {sorted(missing)}.\n"
        f"Either they are retired, or this account has no entitlement for "
        f"them — check the provider's deprecation list before removing an "
        f"entry, because an entitlement gap is specific to one account "
        f"while a retirement affects every user.\n"
        f"If retired: update src/rlmstudio/domain/model_catalog.py and any "
        f"default that names them.\n"
        f"Currently available (first 15): {sorted(available)[:15]}"
    )


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="needs OPENAI_API_KEY; run before cutting a release",
)
def test_every_catalog_openai_model_is_still_offered() -> None:
    available = _openai_model_names(os.environ["OPENAI_API_KEY"])
    catalogued = {m.name for m in OPENAI_MODELS}

    missing = catalogued - available
    assert not missing, (
        f"the catalog offers models this account cannot use: {sorted(missing)}.\n"
        f"Either retired, or no entitlement on this account — OpenAI gates "
        f"some models by usage tier, so confirm against the deprecation "
        f"list before removing an entry.\n"
        f"If retired: update src/rlmstudio/domain/model_catalog.py."
    )


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="needs ANTHROPIC_API_KEY; run before cutting a release",
)
def test_configured_providers_name_models_the_account_offers() -> None:
    """The user's own saved providers, not just what we ship.

    The 2026-10-28 incident came from persisted config rather than from a
    shipped default: a saved provider still named a model retired months
    earlier, and nothing re-checked it. This covers that half.
    """
    from rlmstudio.server.dependencies import get_state

    available = _anthropic_model_names(os.environ["ANTHROPIC_API_KEY"])
    stale = [
        (lp.name, lp.model)
        for lp in get_state().config.llm_providers
        if lp.backend == "anthropic" and lp.model and lp.model not in available
    ]

    assert not stale, (
        f"saved LLM providers name models this account cannot use: {stale}.\n"
        f"Update them in Settings → Providers."
    )
