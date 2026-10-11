# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0
"""
RunProfile & RunProfileStore — named configuration profiles.

A RunProfile captures "how I want the system to behave":
  execution strategy, sampling params, RLM budget, RAG params, etc.

Profiles are persisted as a JSON array in ``<state dir>/profiles.json``.
"""

from __future__ import annotations

import importlib.resources
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rlmstudio.application.sandbox_vars import (
    MODE_DIRECT,
    MODE_RAG,
    MODE_RLM,
    RLM_DEFAULT_WALL_CLOCK_BUDGET_SECONDS,
)
from rlmstudio.branding import state_dir


@dataclass
class RunProfile:
    """A named set of execution settings."""

    name: str
    """Profile display name, e.g. 'Fast & cheap'."""

    # -- strategy --
    run_mode: str = "single"
    """'single' or 'compare'."""

    strategy: str = MODE_DIRECT
    """'direct', 'rlm', or 'rag' (only used in single mode)."""

    default_provider: str | None = None
    """Provider key used in single mode."""

    providers_enabled: list[str] = field(default_factory=list)
    """Provider keys used in compare mode."""

    # -- sampling --
    temperature: float = 0.7
    top_p: float = 1.0
    max_output_tokens: int = 2000

    # -- RLM --
    max_steps: int = 16
    rlm_timeout_seconds: int = RLM_DEFAULT_WALL_CLOCK_BUDGET_SECONDS

    # -- RAG --
    rag_top_k: int = 5
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 200
    rag_embedding_model: str = "text-embedding-3-small"

    # -- system prompt --
    system_prompt_mode: str = "default"
    """'default' (use a named template) or 'custom' (user-supplied text)."""

    system_prompt_template: str = "Default"
    """Name of the built-in template to use when mode is 'default'."""

    system_prompt_custom: dict[str, str] | None = None
    """Per-mode user-supplied prompts when mode is 'custom'. Keys: 'direct', 'rag', 'rlm'."""

    # -- key policy --
    key_policy: str | None = None
    """'env', 'file', or 'keyring'. None = use session default."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunProfile:
        # Filter out unknown keys for forward-compat
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        # Backward compat: migrate old single-string system_prompt_custom to per-mode dict
        custom = filtered.get("system_prompt_custom")
        if isinstance(custom, str):
            # Old format was a single string; apply it to all modes
            filtered["system_prompt_custom"] = {
                MODE_DIRECT: custom,
                MODE_RAG: custom,
                MODE_RLM: custom,
            }
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------

BUILTIN_PROFILES: list[RunProfile] = [
    RunProfile(
        name="Fast & cheap",
        strategy=MODE_DIRECT,
        temperature=0.5,
        max_output_tokens=1000,
        max_steps=8,
        rlm_timeout_seconds=RLM_DEFAULT_WALL_CLOCK_BUDGET_SECONDS,
    ),
    RunProfile(
        name="Accurate",
        strategy=MODE_DIRECT,
        temperature=0.2,
        max_output_tokens=4000,
        top_p=0.95,
    ),
    RunProfile(
        name="RLM deep",
        strategy=MODE_RLM,
        max_steps=32,
        rlm_timeout_seconds=RLM_DEFAULT_WALL_CLOCK_BUDGET_SECONDS,
        temperature=0.4,
        max_output_tokens=4000,
    ),
]

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

_DEFAULT_PROFILES_PATH = state_dir() / "profiles.json"


class RunProfileStore:
    """Load / save ``RunProfile`` objects from a JSON file."""

    def __init__(self, path: Path | None = None):
        self._path = Path(path) if path else _DEFAULT_PROFILES_PATH

    # -- read --------------------------------------------------------------

    def list_profiles(self) -> list[RunProfile]:
        """Return all profiles (builtins + user-saved)."""
        return BUILTIN_PROFILES + self._load_user()

    def get_profile(self, name: str) -> RunProfile | None:
        for p in self.list_profiles():
            if p.name == name:
                return p
        return None

    # -- write -------------------------------------------------------------

    def save_profile(self, profile: RunProfile) -> None:
        """Save or overwrite a user profile by name."""
        profiles = self._load_user()
        profiles = [p for p in profiles if p.name != profile.name]
        profiles.append(profile)
        self._write_user(profiles)

    def delete_profile(self, name: str) -> bool:
        """Delete a user profile. Returns False if name is a builtin."""
        if any(p.name == name for p in BUILTIN_PROFILES):
            return False
        profiles = self._load_user()
        before = len(profiles)
        profiles = [p for p in profiles if p.name != name]
        if len(profiles) < before:
            self._write_user(profiles)
            return True
        return False

    # -- internal ----------------------------------------------------------

    def _load_user(self) -> list[RunProfile]:
        if not self._path.exists():
            return []
        try:
            with open(self._path) as f:
                items = json.load(f)
            return [RunProfile.from_dict(d) for d in items]
        except (OSError, json.JSONDecodeError):
            return []

    def _write_user(self, profiles: list[RunProfile]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump([p.to_dict() for p in profiles], f, indent=2)


# ---------------------------------------------------------------------------
# Custom system prompt persistence  (<state dir>/system_prompt_custom.json)
# ---------------------------------------------------------------------------

_DEFAULT_CUSTOM_PROMPTS_PATH = state_dir() / "system_prompt_custom.json"


def load_custom_prompts(path: Path | None = None) -> dict[str, str]:
    """Load user custom prompts from JSON file.

    Returns dict with keys 'direct', 'rag', 'rlm' (may be empty strings).
    """
    p = Path(path) if path else _DEFAULT_CUSTOM_PROMPTS_PATH
    if not p.exists():
        return {MODE_DIRECT: "", MODE_RAG: "", MODE_RLM: ""}
    try:
        with open(p) as f:
            data = json.load(f)
        # Ensure all keys exist
        return {
            MODE_DIRECT: data.get(MODE_DIRECT, ""),
            MODE_RAG: data.get(MODE_RAG, ""),
            MODE_RLM: data.get(MODE_RLM, ""),
        }
    except (OSError, json.JSONDecodeError):
        return {MODE_DIRECT: "", MODE_RAG: "", MODE_RLM: ""}


def save_custom_prompts(
    prompts: dict[str, str],
    path: Path | None = None,
) -> None:
    """Save user custom prompts to JSON file."""
    p = Path(path) if path else _DEFAULT_CUSTOM_PROMPTS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(prompts, f, indent=2)


# ---------------------------------------------------------------------------
# System prompt templates  (loaded from rlmstudio/prompts/system_prompt_templates.json)
# ---------------------------------------------------------------------------

_TEMPLATES_FILE = "system_prompt_templates.json"

_templates_cache: dict[str, dict[str, Any]] | None = None


def _load_system_prompt_templates() -> dict[str, dict[str, Any]]:
    """Load system prompt templates from the JSON data file.

    The file lives in the ``rlmstudio.prompts`` package so it is included
    in both editable installs and built wheels.
    """
    global _templates_cache
    if _templates_cache is not None:
        return _templates_cache

    try:
        files = importlib.resources.files("rlmstudio.prompts")
        raw = (files / _TEMPLATES_FILE).read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, TypeError):
        # Fallback: resolve relative to this file
        fallback = Path(__file__).resolve().parents[2] / "prompts" / _TEMPLATES_FILE
        raw = fallback.read_text(encoding="utf-8")

    _templates_cache = json.loads(raw)
    return _templates_cache


def get_system_prompt_templates() -> dict[str, dict[str, Any]]:
    """Public accessor for the template catalog."""
    return _load_system_prompt_templates()


def reload_system_prompt_templates() -> dict[str, dict[str, Any]]:
    """Force-reload templates from disk (e.g. after editing the JSON).

    Also refreshes the module-level ``SYSTEM_PROMPT_TEMPLATES`` dict so
    that callers who imported the constant see the updated values, and
    clears the prompt-rendering caches in ``rlmstudio.prompts.templates``
    so that Direct/RAG/RLM prompts pick up the new text.
    """
    global _templates_cache, SYSTEM_PROMPT_TEMPLATES, SYSTEM_PROMPT_TEMPLATE_NAMES
    _templates_cache = None
    fresh = _load_system_prompt_templates()
    SYSTEM_PROMPT_TEMPLATES.clear()
    SYSTEM_PROMPT_TEMPLATES.update(fresh)
    SYSTEM_PROMPT_TEMPLATE_NAMES.clear()
    SYSTEM_PROMPT_TEMPLATE_NAMES.extend(fresh.keys())

    # Clear the prompt-rendering caches so subsequent calls use the new templates.
    from rlmstudio.prompts.templates import _template_cache, _yaml_doc_cache

    _template_cache.clear()
    _yaml_doc_cache.clear()

    return fresh


def get_system_prompt_template_names() -> list[str]:
    """Return ordered list of template names."""
    return list(get_system_prompt_templates().keys())


# Module-level constants so existing ``from profile_store import SYSTEM_PROMPT_TEMPLATES``
# continues to work.  The JSON is small and loaded once at import time.
SYSTEM_PROMPT_TEMPLATES: dict[str, dict[str, Any]] = _load_system_prompt_templates()
SYSTEM_PROMPT_TEMPLATE_NAMES: list[str] = list(SYSTEM_PROMPT_TEMPLATES.keys())


def resolve_system_prompt(
    mode: str,
    session_state: Any = None,
) -> str | None:
    """Resolve the active system prompt for a given execution *mode*.

    Resolution order:
      1. Session-state overrides (``system_prompt_mode``, ``system_prompt_template``,
         ``system_prompt_custom``)
      2. ``None`` — meaning "let the strategy / RLM use its own built-in default"

    Args:
        mode: One of ``"direct"``, ``"rag"``, ``"rlm"``.
        session_state: Streamlit session state (or any object with attribute access).

    Returns:
        The prompt string to pass to the strategy constructor, or ``None``
        when the built-in default should be used.
    """
    prompt_mode = "default"
    template_name = "Default"
    custom_prompts: dict[str, str] | None = None

    if session_state is not None:
        prompt_mode = getattr(session_state, "system_prompt_mode", None) or "default"
        template_name = getattr(session_state, "system_prompt_template", None) or "Default"
        custom_prompts = getattr(session_state, "system_prompt_custom", None)

    if prompt_mode == "custom" and custom_prompts:
        # Per-mode custom prompts: look up the specific mode
        if isinstance(custom_prompts, dict):
            text = custom_prompts.get(mode, "")
            if text:
                return text
        # Backward compat: if somehow still a string, use it
        elif isinstance(custom_prompts, str) and custom_prompts:
            return custom_prompts
        # Empty custom → fall through to template

    # "default" mode → look up the named template
    tpl = SYSTEM_PROMPT_TEMPLATES.get(template_name)
    if tpl is None:
        return None  # unknown template → fall back to built-in
    return tpl.get(mode)  # may be None → strategy default
