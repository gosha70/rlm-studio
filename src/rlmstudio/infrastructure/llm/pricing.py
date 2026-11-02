"""Completion pricing for read-only views.

The run path prices through the adapter (``LLMPort.get_completion_cost`` /
``get_pricing``). The trace view has no adapter instance — only the model id
and the per-step token counts — so this helper exposes the same LiteLLM cost
map without constructing one. It lives in ``infrastructure/llm`` so the
no-SDK-imports-outside-this-package rule still holds.
"""

from __future__ import annotations


def completion_cost(model: str | None, prompt_tokens: int, completion_tokens: int) -> float | None:
    """Return the USD cost of one completion, or ``None`` when unpriced.

    ``None`` means LiteLLM's cost map does not know the model: the trace
    shows "Unavailable" rather than a fabricated ``$0.0000``.  A known model
    is priced from its published per-token rates; a row with no tokens of its
    own (``subcall``, runtime fingerprint) is a genuine ``0.0`` regardless.

    Cache-discounted rates are not applied, matching the run path, which
    prices every input token at the model's input rate.
    """
    if prompt_tokens <= 0 and completion_tokens <= 0:
        return 0.0
    if not model:
        return None

    try:
        # Lazy import: pricing is only needed when a trace is rendered, and
        # importing litellm at module load slows every server start.
        import litellm

        prompt_cost, response_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        return float(prompt_cost) + float(response_cost)
    except Exception:
        # An unknown model (or an unpriceable one) is reported as unavailable,
        # never as free.  See issue #94 and the #101 pricing fix.
        return None
