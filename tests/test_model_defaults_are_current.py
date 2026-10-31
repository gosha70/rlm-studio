"""Guard against shipping a default that names a retired model.

A live-provider check on 2026-10-28 failed six requests against
``claude-sonnet-4-20250514``, which Anthropic has retired. That came from
persisted config, but the audit it triggered found the same class of problem
compiled into the package: the legacy ``llm/`` clients and the
``ModelConfig`` presets carried their own hardcoded Claude 3 identifiers,
all retired, so ``provider="anthropic"`` on the public client failed with
``not_found_error`` out of the box.

The fix moved the model list into ``domain/model_catalog.py`` — the innermost
layer, so ``llm/`` and ``core/`` no longer reach into UI-owned data for it —
and declared each default there. These tests pin that relationship: a default
that falls out of the catalog, or a price the catalog does not know, fails
here rather than at a user's first run.
"""

from __future__ import annotations

import inspect

import pytest

from rlmstudio.domain.model_catalog import (
    ANTHROPIC_MODELS,
    DEFAULT_ANTHROPIC_CHEAP_MODEL,
    DEFAULT_ANTHROPIC_HIGH_QUALITY_MODEL,
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
        (
            "DEFAULT_ANTHROPIC_HIGH_QUALITY_MODEL",
            DEFAULT_ANTHROPIC_HIGH_QUALITY_MODEL,
            ANTHROPIC_MODELS,
        ),
        ("DEFAULT_OPENAI_MODEL", DEFAULT_OPENAI_MODEL, OPENAI_MODELS),
    ],
)
def test_each_default_is_a_model_the_catalog_offers(
    label: str, default: str, catalog: list
) -> None:
    names = {m.name for m in catalog}
    assert default in names, f"{label} ({default!r}) is not offered by the catalog"


def test_each_default_has_the_tier_it_is_named_for() -> None:
    """Tier is explicit metadata, so assert on it rather than on the name.

    Pinning the tier matters because the catalog's first entry is Opus:
    deriving the balanced default positionally would silently move existing
    callers onto Opus pricing — a surprise cost rather than a retirement fix.
    """
    from rlmstudio.domain.model_catalog import ModelTier

    by_name = {m.name: m for m in ANTHROPIC_MODELS}

    assert by_name[DEFAULT_ANTHROPIC_MODEL].tier is ModelTier.BALANCED
    assert by_name[DEFAULT_ANTHROPIC_CHEAP_MODEL].tier is ModelTier.CHEAP
    assert by_name[DEFAULT_ANTHROPIC_HIGH_QUALITY_MODEL].tier is ModelTier.HIGH_QUALITY


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


# ---------------------------------------------------------------------------
# Pricing: the bug the retirement fix activated
# ---------------------------------------------------------------------------


def test_every_default_has_a_known_price() -> None:
    """A default the pricing path cannot price is how the overcharge happened.

    The legacy client's own table knew only Claude 2 and 3 and fell back to
    Opus 3 for anything else, so the new Sonnet 4.6 default was billed at
    Opus 3 rates — $90 for 1M in + 1M out against a real $18.
    """
    from rlmstudio.domain.model_catalog import price_per_1k

    for default in (
        DEFAULT_ANTHROPIC_MODEL,
        DEFAULT_ANTHROPIC_CHEAP_MODEL,
        DEFAULT_ANTHROPIC_HIGH_QUALITY_MODEL,
        DEFAULT_OPENAI_MODEL,
    ):
        assert price_per_1k(default) is not None, f"no price known for {default}"


def test_legacy_client_prices_the_default_from_the_catalog() -> None:
    """The exact overcharge, pinned."""
    from rlmstudio.domain.model_catalog import price_per_1k
    from rlmstudio.llm.anthropic_client import ClaudeClient

    client = ClaudeClient.__new__(ClaudeClient)
    client.model = DEFAULT_ANTHROPIC_MODEL

    input_per_1k, output_per_1k = price_per_1k(DEFAULT_ANTHROPIC_MODEL)  # type: ignore[misc]
    expected = 1000 * input_per_1k + 1000 * output_per_1k

    assert ClaudeClient.calculate_cost(client, 1_000_000, 1_000_000) == pytest.approx(expected)


def test_unknown_model_reports_no_cost_rather_than_a_guess() -> None:
    """Substituting another model's rate is what produced the overcharge.

    0.0 means "not priced" to callers; a confidently wrong figure does not.
    """
    from rlmstudio.llm.anthropic_client import ClaudeClient

    client = ClaudeClient.__new__(ClaudeClient)
    client.model = "some-model-the-catalog-has-never-heard-of"

    assert ClaudeClient.calculate_cost(client, 1_000_000, 1_000_000) == 0.0


def test_ui_catalog_offers_exactly_the_domain_catalog() -> None:
    """The UI list derives from the domain one; pin that nothing is lost.

    Moving the lists into the domain layer, I truncated OpenAI from seven
    entries to three. Nothing failed at the boundary — the metrics collector
    simply priced `gpt-4` at zero, three layers away. Comparing the two lists
    catches a dropped or mispriced entry at the source.
    """
    from rlmstudio.domain.model_catalog import ANTHROPIC_MODELS as DOMAIN_ANTHROPIC
    from rlmstudio.domain.model_catalog import OPENAI_MODELS as DOMAIN_OPENAI
    from rlmstudio.ui.data.providers_catalog import ANTHROPIC_MODELS as UI_ANTHROPIC
    from rlmstudio.ui.data.providers_catalog import OPENAI_MODELS as UI_OPENAI

    for label, domain, ui in (
        ("anthropic", DOMAIN_ANTHROPIC, UI_ANTHROPIC),
        ("openai", DOMAIN_OPENAI, UI_OPENAI),
    ):
        assert [m.name for m in domain] == [m.name for m in ui], f"{label} names differ"
        for d, u in zip(domain, ui, strict=True):
            assert d.input_cost_per_1k == u.input_cost_per_1k, f"{label} {d.name} input price"
            assert d.output_cost_per_1k == u.output_cost_per_1k, f"{label} {d.name} output price"


# ---------------------------------------------------------------------------
# Prices, pinned against the published table rather than against themselves
# ---------------------------------------------------------------------------

# Taken from https://platform.claude.com/docs/en/about-claude/pricing on
# 2026-10-31, converted from per-million to per-thousand tokens.
#
# These literals are the point. Every other test compares the pricing path
# against the catalog, so a wrong figure in the catalog agreed with itself and
# passed: Opus was recorded at Sonnet's rate and Haiku at $8/$24 — dearer than
# Sonnet, despite being the cheap tier. Pinning the published numbers here is
# what makes the catalog answerable to something outside itself.
PUBLISHED_ANTHROPIC_PRICES_PER_1K = {
    "claude-opus-4-6": (0.005, 0.025),
    "claude-sonnet-4-6": (0.003, 0.015),
    "claude-haiku-4-5-20251001": (0.001, 0.005),
}


@pytest.mark.parametrize(("model", "expected"), sorted(PUBLISHED_ANTHROPIC_PRICES_PER_1K.items()))
def test_catalog_price_matches_the_published_rate(
    model: str, expected: tuple[float, float]
) -> None:
    from rlmstudio.domain.model_catalog import price_per_1k

    assert price_per_1k(model) == expected, (
        f"{model} is priced {price_per_1k(model)} but the published rate is "
        f"{expected}. If Anthropic changed it, update both this literal and "
        f"the catalog, and note the date."
    )


def test_cheaper_tiers_actually_cost_less() -> None:
    """A sanity check the recorded figures failed.

    Haiku was priced above Sonnet, which is the kind of thing that should be
    obvious on sight and was not. Ordering by tier catches a transposed or
    misplaced figure without needing to know the right number.
    """
    from rlmstudio.domain.model_catalog import ModelTier, price_per_1k

    by_tier = {m.tier: m.name for m in ANTHROPIC_MODELS}

    def total(name: str) -> float:
        rates = price_per_1k(name)
        assert rates is not None
        return sum(rates)

    cheap = total(by_tier[ModelTier.CHEAP])
    balanced = total(by_tier[ModelTier.BALANCED])
    high = total(by_tier[ModelTier.HIGH_QUALITY])

    assert cheap < balanced < high, (
        f"tier pricing is out of order: cheap={cheap}, balanced={balanced}, high_quality={high}"
    )
