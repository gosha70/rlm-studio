"""Named constants for sandbox variable names and config-extra keys.

These identifiers cross module boundaries (use cases set them,
infrastructure skips them on return), so they live here as the single
source of truth rather than as magic strings scattered across files.
"""

from typing import Literal, TypeAlias

# ---------------------------------------------------------------------------
# Sandbox variable names — bound via sandbox.set_variable(), read by model
# code inside the subprocess, and excluded from the child → parent return
# to avoid O(payload-size) round-trips.
# ---------------------------------------------------------------------------

SANDBOX_VAR_DOCUMENT = "P"
"""The full document content variable.  Set by RunRLMUseCase."""

SANDBOX_VAR_FILE_INDEX = "_FILE_INDEX"
"""Multi-file index (forward reservation).  Excluded from return."""

SANDBOX_VAR_HISTORY = "history"
"""Conversation history list.  Set by RunRLMUseCase when conversation
memory is enabled; the model reads it via ``print(history[-1])``."""

# The set of variable names that should NOT round-trip child → parent
# after each sandbox execution.  Used by both the child-side
# ``_child_worker`` (skip_return) and the parent-side ``execute``
# (skip_keys) in ``subprocess_sandbox.py``.
SANDBOX_SKIP_RETURN_VARS = frozenset(
    {SANDBOX_VAR_DOCUMENT, SANDBOX_VAR_FILE_INDEX, SANDBOX_VAR_HISTORY}
)

# ---------------------------------------------------------------------------
# RunConfigDTO.extra keys — carry data from the routing layer to use cases
# ---------------------------------------------------------------------------

EXTRA_KEY_HISTORY_VARIABLE = "history_variable"
"""Key in ``RunConfigDTO.extra`` that carries the ``history`` list from
``_prepare_history_context`` in ``chat.py`` to ``RunRLMUseCase``."""

EXTRA_KEY_HISTORY_MESSAGES = "history_messages"
"""Key in ``RunConfigDTO.extra`` that carries prior-turn chat messages
(native user/assistant pairs) for Direct, Compare, and RAG modes."""

# ---------------------------------------------------------------------------
# History context path identifiers — returned in history_info["path"]
# ---------------------------------------------------------------------------

HISTORY_PATH_DISABLED = "disabled"
HISTORY_PATH_EMPTY = "empty"
HISTORY_PATH_INPROMPT = "inprompt"
HISTORY_PATH_REPL_VARIABLE = "repl_variable"

# ---------------------------------------------------------------------------
# Execution mode identifiers
# ---------------------------------------------------------------------------

MODE_DIRECT = "direct"
MODE_RLM = "rlm"
MODE_RAG = "rag"
MODE_COMPARE = "compare"
MODE_AUTO = "auto"
MODE_RLM_OFFICIAL = "rlm_official"
"""The paper authors' reference implementation (``rlms``) run as an engine."""

ExecutionMode: TypeAlias = Literal["direct", "rlm", "rag", "rlm_official"]
"""Modes a Chat Provider or a Compare-matrix slot executes in.

The literal is spelled here, next to the ``MODE_*`` constants, so every
request model and use case imports one alias instead of repeating the set.
"""

ChatMode: TypeAlias = Literal["auto", "rlm", "direct", "rag", "compare", "rlm_official"]
"""Modes accepted by ``POST /api/chat``.  ``auto`` never resolves to ``rlm_official``."""

# Modes that use RLM internally (RLM run config applies)
MODES_RLM_INTERNAL = frozenset({MODE_RLM, MODE_AUTO, MODE_COMPARE, MODE_RLM_OFFICIAL})

# ---------------------------------------------------------------------------
# Mode groupings — which execution modes use which history path
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# RLM execution defaults
# ---------------------------------------------------------------------------

# Default configuration values — single source of truth so changing a
# default doesn't require scanning every file in the repo.

RLM_DEFAULT_MAX_STEPS = 16
RLM_DEFAULT_WALL_CLOCK_BUDGET_SECONDS = 600
"""Total wall-clock budget for an entire RLM execution (all steps combined).
Not the per-step LLM call timeout.  600s = 10 minutes, enough for ~6-7
steps on slow local models at ~90s/step."""
RLM_DEFAULT_REPEAT_LIMIT = 2
RLM_DEFAULT_STALL_LIMIT = 3
"""Consecutive no-progress steps before the circuit-breaker fires."""
RLM_DEFAULT_NUDGE_AT_FRACTION = 0.4
RLM_DEFAULT_MAX_TOKENS_BUDGET = 50000
RLM_DEFAULT_MAX_COST_USD = 2.0
RLM_DEFAULT_MAX_RECURSION_DEPTH = 5

RUNTIME_DEFAULT_TEMPERATURE = 0.7
RUNTIME_DEFAULT_TOP_P = 1.0
RUNTIME_DEFAULT_MAX_OUTPUT_TOKENS = 4096
RUNTIME_DEFAULT_TIMEOUT_SECONDS = 120

# ---------------------------------------------------------------------------
# Trace entry keys — used in run_rlm.py, run_direct.py, run_rag.py trace
# dicts and read by chat.py for telemetry recording.  NOT the same as
# LLM chat message keys ("role"/"content") which are API conventions.
# ---------------------------------------------------------------------------

TRACE_KEY_STEP = "step"
TRACE_KEY_SEQ = "seq"
TRACE_KEY_ROLE = "role"
TRACE_KEY_CONTENT = "content"
TRACE_KEY_CODE = "code"
TRACE_KEY_INPUT_TOKENS = "input_tokens"
TRACE_KEY_OUTPUT_TOKENS = "output_tokens"
TRACE_KEY_ELAPSED_SECONDS = "elapsed_seconds"
TRACE_KEY_MODEL = "model"
TRACE_KEY_MODE = "mode"
TRACE_KEY_CLAMP = "clamp"

TRACE_KEY_RECURSION_DEPTH = "recursion_depth"
"""Depth of a nested sub-call entry (``1`` = a child query spawned from the root REPL)."""

RESULT_KEY_ENGINE_VERSION = "engine_version"
"""``RunResultDTO.metadata`` key: version of the third-party engine that ran."""
RESULT_KEY_COST_KNOWN = "cost_known"
"""``RunResultDTO.metadata`` key: ``False`` when no price was reported or derivable, so ``total_cost`` is 0 by ignorance, not by measurement."""
RESULT_KEY_ENGINE_NOTES = "engine_notes"
"""``RunResultDTO.metadata`` key: user-facing caveats about the engine run (list of strings)."""

MODES_INPROMPT = frozenset({MODE_DIRECT, MODE_COMPARE, MODE_RAG})
"""Modes that carry history as native user/assistant chat messages."""

MODES_REPL_VARIABLE = frozenset({MODE_RLM, MODE_AUTO})
"""Modes that carry history as a sandbox ``history`` Python variable."""

# ---------------------------------------------------------------------------
# Ephemeral Chat Provider naming
# ---------------------------------------------------------------------------

EPHEMERAL_CP_PREFIX = "[compare] "
"""Name prefix for temporary CPs created by the compare-matrix endpoint.

Used to:
- Identify ephemeral CPs for dedup and cleanup
- Filter them from dashboard metrics and CP listings
- Strip the prefix when displaying names
"""
