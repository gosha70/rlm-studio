"""Canonical model catalog: names, tiers and prices, with no dependencies.

One list, in the innermost layer, so every caller shares it. Before this
existed the same facts were duplicated across the UI catalog, the legacy
``llm/`` clients, the ``core/`` presets and two pricing tables, and they
drifted: five shipped defaults named Claude 3 models Anthropic had retired,
so ``provider="anthropic"`` failed with ``not_found_error`` out of the box,
while the pricing tables charged a current model at a retired model's rate.

Design notes, both load-bearing:

* **Defaults are declared, not derived.** ``DEFAULT_ANTHROPIC_MODEL`` is a
  literal checked against the catalog by a test, rather than a
  ``next(m for m in ... if "sonnet" in m.name)`` evaluated at import. A
  derived default raises at *import* time if the matching tier ever
  disappears, which would take down every ``rlmstudio.llm`` import with a
  bare ``StopIteration``. A declared default that falls out of the catalog
  fails one test instead.

* **Tier is explicit metadata, not a substring of the name.** Matching
  ``"sonnet" in name`` guesses at a vendor's naming convention; ``tier``
  states the intent, so a preset that wants the high-quality model keeps
  getting it when the vendor renames things.

Retiring a model is one deletion here, plus whatever default pointed at it.
``tests/test_model_defaults_are_current.py`` fails if a default is left
dangling.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "ANTHROPIC_MODELS",
    "CatalogModel",
    "DEFAULT_ANTHROPIC_CHEAP_MODEL",
    "DEFAULT_ANTHROPIC_HIGH_QUALITY_MODEL",
    "DEFAULT_ANTHROPIC_MODEL",
    "DEFAULT_OPENAI_MODEL",
    "ModelTier",
    "OPENAI_MODELS",
    "price_per_1k",
]


class ModelTier(str, Enum):
    """What a model is for, independent of how the vendor names it."""

    HIGH_QUALITY = "high_quality"
    BALANCED = "balanced"
    CHEAP = "cheap"


@dataclass(frozen=True)
class CatalogModel:
    """A model this project knows how to name and price.

    Costs are USD per 1,000 tokens, matching how the UI displays them.
    """

    name: str
    input_cost_per_1k: float
    output_cost_per_1k: float
    tier: ModelTier


# Prices verified against https://platform.claude.com/docs/en/about-claude/pricing
# on 2026-10-31. Figures there are per million tokens; these are per thousand,
# so divide by 1000: $5/MTok -> 0.005.
#
# Two of these were wrong before that check. Opus was recorded at Sonnet's
# rate, understating it; Haiku was recorded at $8/$24 — dearer than Sonnet,
# which should have been the tell, since it is the cheap tier. Nothing caught
# it because the only test compared the pricing path against this same table,
# so wrong source data agreed with itself. tests/test_model_defaults_are_current.py
# now pins these three rates as literals taken from the page above.
ANTHROPIC_MODELS: tuple[CatalogModel, ...] = (
    CatalogModel("claude-opus-4-6", 0.005, 0.025, ModelTier.HIGH_QUALITY),
    CatalogModel("claude-sonnet-4-6", 0.003, 0.015, ModelTier.BALANCED),
    CatalogModel("claude-haiku-4-5-20251001", 0.001, 0.005, ModelTier.CHEAP),
)

OPENAI_MODELS: tuple[CatalogModel, ...] = (
    CatalogModel("gpt-4o", 0.005, 0.015, ModelTier.BALANCED),
    CatalogModel("gpt-4o-mini", 0.00015, 0.0006, ModelTier.CHEAP),
    CatalogModel("gpt-4-turbo", 0.01, 0.03, ModelTier.BALANCED),
    CatalogModel("gpt-4", 0.03, 0.06, ModelTier.HIGH_QUALITY),
    CatalogModel("gpt-3.5-turbo", 0.0005, 0.0015, ModelTier.CHEAP),
    CatalogModel("o1", 0.015, 0.06, ModelTier.HIGH_QUALITY),
    CatalogModel("o1-mini", 0.003, 0.012, ModelTier.BALANCED),
)

# Declared, not derived — see the module docstring.
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_ANTHROPIC_HIGH_QUALITY_MODEL = "claude-opus-4-6"
DEFAULT_ANTHROPIC_CHEAP_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_OPENAI_MODEL = "gpt-4o"

_BY_NAME: dict[str, CatalogModel] = {
    model.name: model for model in (*ANTHROPIC_MODELS, *OPENAI_MODELS)
}


def price_per_1k(model: str) -> tuple[float, float] | None:
    """Return ``(input, output)`` USD per 1,000 tokens, or ``None``.

    ``None`` rather than a fallback price: callers that silently substituted
    another model's rate reported a current Sonnet at Claude 3 Opus rates —
    five times the real cost — and nothing surfaced it. An unknown model has
    no known price, and saying so lets the caller report that honestly.
    """
    found = _BY_NAME.get(model)
    if found is None:
        return None
    return found.input_cost_per_1k, found.output_cost_per_1k
