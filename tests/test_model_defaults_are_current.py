"""Guard against shipping a default that names a retired model.

A live-provider check on 2026-10-28 failed six requests against
``claude-sonnet-4-20250514``, which Anthropic has retired. That came from
persisted config, but the audit it triggered found the same class of problem
compiled into the package: the legacy ``llm/`` clients and the
``ModelConfig`` presets carried their own hardcoded Claude 3 identifiers,
all retired, so ``provider="anthropic"`` on the public client failed with
``not_found_error`` out of the box.

The fix was to derive those defaults from ``providers_catalog``, which is the
list the UI already offers. These tests pin that relationship: a default that
is not in the catalog fails here rather than at a user's first run.
"""

from __future__ import annotations

import inspect

import pytest

from rlmstudio.ui.data.providers_catalog import (
    ANTHROPIC_MODELS,
    DEFAULT_ANTHROPIC_CHEAP_MODEL,
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OPENAI_MODEL,
    OPENAI_MODELS,
)

# Identifiers Anthropic has retired. Not exhaustive — it is the set this
# package actually shipped, which is what a regression would reintroduce.
RETIRED_ANTHROPIC_MODELS = frozenset(
    {
        "claude-3-sonnet-20240229",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-sonnet-4-20250514",
    }
)


def test_catalog_offers_no_retired_model() -> None:
    """The catalog is what the UI lists and what every default derives from."""
    offered = {m.name for m in ANTHROPIC_MODELS}
    assert not (offered & RETIRED_ANTHROPIC_MODELS), (
        f"the providers catalog offers retired models: {sorted(offered & RETIRED_ANTHROPIC_MODELS)}"
    )


@pytest.mark.parametrize(
    ("label", "default", "catalog"),
    [
        ("DEFAULT_ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL, ANTHROPIC_MODELS),
        ("DEFAULT_ANTHROPIC_CHEAP_MODEL", DEFAULT_ANTHROPIC_CHEAP_MODEL, ANTHROPIC_MODELS),
        ("DEFAULT_OPENAI_MODEL", DEFAULT_OPENAI_MODEL, OPENAI_MODELS),
    ],
)
def test_each_default_is_a_model_the_catalog_offers(
    label: str, default: str, catalog: list
) -> None:
    names = {m.name for m in catalog}
    assert default in names, f"{label} ({default!r}) is not offered by the catalog"


def test_anthropic_default_keeps_its_tier() -> None:
    """The defaults these replaced were Sonnet- and Haiku-tier.

    Pinning the tier matters because the catalog's first entry is Opus:
    deriving the default positionally would silently move existing callers
    onto Opus pricing, which is a surprise cost rather than a retirement fix.
    """
    assert "sonnet" in DEFAULT_ANTHROPIC_MODEL
    assert "haiku" in DEFAULT_ANTHROPIC_CHEAP_MODEL


def test_legacy_anthropic_client_default_is_current() -> None:
    """`ClaudeClient()` with no model must not name a retired identifier."""
    from rlmstudio.llm.anthropic_client import ClaudeClient

    default = inspect.signature(ClaudeClient.__init__).parameters["model"].default

    assert default not in RETIRED_ANTHROPIC_MODELS, f"retired default: {default}"
    assert default == DEFAULT_ANTHROPIC_MODEL


def test_auto_detection_default_is_current() -> None:
    """`llm/auto.py` picks a model when the caller names only a provider."""
    import rlmstudio.llm.auto as auto

    source = inspect.getsource(auto)

    for retired in RETIRED_ANTHROPIC_MODELS:
        assert f'"{retired}"' not in source, f"auto.py still names retired {retired}"


@pytest.mark.parametrize("provider", ["openai", "anthropic", "mixed"])
def test_cost_optimized_presets_name_no_retired_model(provider: str) -> None:
    """These presets are selectable, so a retired id here is a runtime failure
    for the user who picks it, not a stale comment."""
    from rlmstudio.core.model_config import ModelConfig

    config = ModelConfig.cost_optimized(provider)

    for field in (config.root_model, config.sub_model):
        assert field not in RETIRED_ANTHROPIC_MODELS, (
            f"{provider} preset names retired model {field}"
        )
