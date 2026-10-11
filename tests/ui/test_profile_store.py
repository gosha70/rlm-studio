# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0
"""Tests for RunProfile persistence, migration defaults, and system prompt resolution."""

import json
from types import SimpleNamespace

from rlmstudio.ui.services.profile_store import (
    BUILTIN_PROFILES,
    SYSTEM_PROMPT_TEMPLATE_NAMES,
    SYSTEM_PROMPT_TEMPLATES,
    RunProfile,
    RunProfileStore,
    load_custom_prompts,
    reload_system_prompt_templates,
    resolve_system_prompt,
    save_custom_prompts,
)

# ---------------------------------------------------------------------------
# RunProfile serialization / backward-compat
# ---------------------------------------------------------------------------


class TestRunProfileSerialization:
    def test_round_trip(self):
        p = RunProfile(name="test", strategy="rlm", temperature=0.3)
        d = p.to_dict()
        p2 = RunProfile.from_dict(d)
        assert p2.name == "test"
        assert p2.strategy == "rlm"
        assert p2.temperature == 0.3

    def test_unknown_keys_ignored(self):
        """Forward-compat: extra keys in JSON are silently dropped."""
        data = {"name": "compat", "future_field": 42, "temperature": 0.9}
        p = RunProfile.from_dict(data)
        assert p.name == "compat"
        assert p.temperature == 0.9
        assert not hasattr(p, "future_field")

    def test_missing_system_prompt_fields_default(self):
        """Backward-compat: old profiles without system_prompt fields get defaults."""
        data = {"name": "old_profile", "strategy": "direct"}
        p = RunProfile.from_dict(data)
        assert p.system_prompt_mode == "default"
        assert p.system_prompt_template == "Default"
        assert p.system_prompt_custom is None

    def test_system_prompt_fields_persisted(self):
        """Per-mode custom prompts are persisted as a dict."""
        custom = {"direct": "Be brief.", "rag": "Use context.", "rlm": "Think step by step."}
        p = RunProfile(
            name="custom_prompt",
            system_prompt_mode="custom",
            system_prompt_template="Concise analyst",
            system_prompt_custom=custom,
        )
        d = p.to_dict()
        assert d["system_prompt_mode"] == "custom"
        assert d["system_prompt_custom"] == custom
        p2 = RunProfile.from_dict(d)
        assert p2.system_prompt_mode == "custom"
        assert p2.system_prompt_custom == custom

    def test_system_prompt_custom_backward_compat(self):
        """Old profiles with string system_prompt_custom get migrated to dict."""
        data = {
            "name": "old_custom",
            "system_prompt_mode": "custom",
            "system_prompt_custom": "You are a pirate.",
        }
        p = RunProfile.from_dict(data)
        assert isinstance(p.system_prompt_custom, dict)
        assert p.system_prompt_custom["direct"] == "You are a pirate."
        assert p.system_prompt_custom["rag"] == "You are a pirate."
        assert p.system_prompt_custom["rlm"] == "You are a pirate."


# ---------------------------------------------------------------------------
# RunProfileStore persistence
# ---------------------------------------------------------------------------


class TestRunProfileStore:
    def test_builtins_always_present(self, tmp_path):
        store = RunProfileStore(path=tmp_path / "profiles.json")
        profiles = store.list_profiles()
        builtin_names = {p.name for p in BUILTIN_PROFILES}
        listed_names = {p.name for p in profiles}
        assert builtin_names.issubset(listed_names)

    def test_save_and_load(self, tmp_path):
        store = RunProfileStore(path=tmp_path / "profiles.json")
        p = RunProfile(name="user1", temperature=0.1)
        store.save_profile(p)
        loaded = store.get_profile("user1")
        assert loaded is not None
        assert loaded.temperature == 0.1

    def test_overwrite_profile(self, tmp_path):
        store = RunProfileStore(path=tmp_path / "profiles.json")
        store.save_profile(RunProfile(name="p", temperature=0.5))
        store.save_profile(RunProfile(name="p", temperature=0.9))
        loaded = store.get_profile("p")
        assert loaded.temperature == 0.9

    def test_delete_user_profile(self, tmp_path):
        store = RunProfileStore(path=tmp_path / "profiles.json")
        store.save_profile(RunProfile(name="deleteme"))
        assert store.delete_profile("deleteme") is True
        assert store.get_profile("deleteme") is None

    def test_cannot_delete_builtin(self, tmp_path):
        store = RunProfileStore(path=tmp_path / "profiles.json")
        assert store.delete_profile(BUILTIN_PROFILES[0].name) is False

    def test_load_corrupt_file(self, tmp_path):
        path = tmp_path / "profiles.json"
        path.write_text("not json!")
        store = RunProfileStore(path=path)
        # Should not crash; returns only builtins
        profiles = store.list_profiles()
        assert len(profiles) == len(BUILTIN_PROFILES)

    def test_migration_old_profiles_json(self, tmp_path):
        """Simulate loading an old profiles.json that lacks system_prompt fields."""
        path = tmp_path / "profiles.json"
        old_data = [{"name": "legacy", "strategy": "direct", "temperature": 0.6}]
        path.write_text(json.dumps(old_data))
        store = RunProfileStore(path=path)
        p = store.get_profile("legacy")
        assert p is not None
        assert p.system_prompt_mode == "default"
        assert p.system_prompt_template == "Default"
        assert p.system_prompt_custom is None


# ---------------------------------------------------------------------------
# resolve_system_prompt
# ---------------------------------------------------------------------------


class TestResolveSystemPrompt:
    def test_default_template_returns_prompts(self):
        """'Default' template returns actual prompt text for all modes."""
        state = SimpleNamespace(
            system_prompt_mode="default",
            system_prompt_template="Default",
            system_prompt_custom=None,
        )
        direct = resolve_system_prompt("direct", state)
        rag = resolve_system_prompt("rag", state)
        rlm = resolve_system_prompt("rlm", state)
        assert direct is not None
        assert "helpful" in direct.lower()
        assert rag is not None
        assert "context" in rag.lower()
        assert rlm is not None
        assert "guidance" in rlm.lower()

    def test_named_template_returns_prompt(self):
        state = SimpleNamespace(
            system_prompt_mode="default",
            system_prompt_template="Concise analyst",
            system_prompt_custom=None,
        )
        result = resolve_system_prompt("direct", state)
        assert result is not None
        assert "concise" in result.lower()

    def test_named_template_rlm_returns_prompt(self):
        """Concise analyst has an RLM-specific prompt."""
        state = SimpleNamespace(
            system_prompt_mode="default",
            system_prompt_template="Concise analyst",
            system_prompt_custom=None,
        )
        result = resolve_system_prompt("rlm", state)
        assert result is not None
        assert "fewest steps" in result.lower()

    def test_custom_mode_returns_custom_text(self):
        """Per-mode custom prompts return the correct prompt for each mode."""
        state = SimpleNamespace(
            system_prompt_mode="custom",
            system_prompt_template="Default",
            system_prompt_custom={
                "direct": "You are a pirate.",
                "rag": "You are a scholar.",
                "rlm": "You are a detective.",
            },
        )
        assert resolve_system_prompt("direct", state) == "You are a pirate."
        assert resolve_system_prompt("rag", state) == "You are a scholar."
        assert resolve_system_prompt("rlm", state) == "You are a detective."

    def test_custom_mode_backward_compat_string(self):
        """Backward compat: string custom prompt still works."""
        state = SimpleNamespace(
            system_prompt_mode="custom",
            system_prompt_template="Default",
            system_prompt_custom="You are a pirate.",  # Old single-string format
        )
        # All modes return the same string
        assert resolve_system_prompt("direct", state) == "You are a pirate."
        assert resolve_system_prompt("rag", state) == "You are a pirate."
        assert resolve_system_prompt("rlm", state) == "You are a pirate."

    def test_custom_mode_empty_text_falls_back(self):
        """Custom mode with empty prompts falls back to template."""
        state = SimpleNamespace(
            system_prompt_mode="custom",
            system_prompt_template="Default",
            system_prompt_custom={"direct": "", "rag": "", "rlm": ""},
        )
        # Empty strings are falsy → falls through to Default template lookup
        result = resolve_system_prompt("direct", state)
        assert result is not None
        assert "helpful" in result.lower()

    def test_custom_mode_partial_prompts(self):
        """Custom mode with some empty prompts falls back for those modes."""
        state = SimpleNamespace(
            system_prompt_mode="custom",
            system_prompt_template="Default",
            system_prompt_custom={"direct": "Custom direct.", "rag": "", "rlm": ""},
        )
        # direct has custom, rag/rlm fall back to template
        assert resolve_system_prompt("direct", state) == "Custom direct."
        assert "context" in resolve_system_prompt("rag", state).lower()
        assert "guidance" in resolve_system_prompt("rlm", state).lower()

    def test_no_session_state(self):
        """No session state → Default template prompts."""
        result_direct = resolve_system_prompt("direct", None)
        result_rag = resolve_system_prompt("rag", None)
        assert result_direct is not None
        assert "helpful" in result_direct.lower()
        assert result_rag is not None
        assert "context" in result_rag.lower()

    def test_unknown_template_returns_none(self):
        state = SimpleNamespace(
            system_prompt_mode="default",
            system_prompt_template="NonExistent",
            system_prompt_custom=None,
        )
        assert resolve_system_prompt("direct", state) is None

    def test_all_templates_have_required_keys(self):
        for name, tpl in SYSTEM_PROMPT_TEMPLATES.items():
            assert "description" in tpl, f"Template '{name}' missing description"
            for mode in ("direct", "rag", "rlm"):
                assert mode in tpl, f"Template '{name}' missing key '{mode}'"

    def test_all_templates_have_nonempty_prompts(self):
        """Every template should provide actual prompt text for all modes."""
        for name, tpl in SYSTEM_PROMPT_TEMPLATES.items():
            for mode in ("direct", "rag", "rlm"):
                assert tpl[mode] is not None, f"Template '{name}' has None for '{mode}'"
                assert len(tpl[mode]) > 20, f"Template '{name}' prompt for '{mode}' is too short"

    def test_template_names_list_matches_dict(self):
        assert SYSTEM_PROMPT_TEMPLATE_NAMES == list(SYSTEM_PROMPT_TEMPLATES.keys())


# ---------------------------------------------------------------------------
# Custom prompts persistence (load_custom_prompts / save_custom_prompts)
# ---------------------------------------------------------------------------


class TestCustomPromptsPersistence:
    def test_load_nonexistent_file(self, tmp_path):
        """Loading from nonexistent file returns empty prompts."""
        path = tmp_path / "nonexistent.json"
        result = load_custom_prompts(path)
        assert result == {"direct": "", "rag": "", "rlm": ""}

    def test_save_and_load(self, tmp_path):
        """Saved custom prompts can be loaded back."""
        path = tmp_path / "custom.json"
        prompts = {
            "direct": "Direct prompt.",
            "rag": "RAG prompt.",
            "rlm": "RLM prompt.",
        }
        save_custom_prompts(prompts, path)
        loaded = load_custom_prompts(path)
        assert loaded == prompts

    def test_load_partial_file(self, tmp_path):
        """Loading a file with missing keys fills in defaults."""
        path = tmp_path / "partial.json"
        path.write_text('{"direct": "Only direct."}')
        result = load_custom_prompts(path)
        assert result["direct"] == "Only direct."
        assert result["rag"] == ""
        assert result["rlm"] == ""

    def test_load_corrupt_file(self, tmp_path):
        """Corrupt JSON returns empty prompts."""
        path = tmp_path / "corrupt.json"
        path.write_text("not valid json!")
        result = load_custom_prompts(path)
        assert result == {"direct": "", "rag": "", "rlm": ""}


class TestReloadSystemPromptTemplates:
    """Tests for reload_system_prompt_templates()."""

    def test_reload_returns_templates(self):
        """reload returns a non-empty dict of templates."""
        result = reload_system_prompt_templates()
        assert isinstance(result, dict)
        assert "Default" in result

    def test_reload_clears_prompt_caches(self):
        """reload clears the prompt-rendering caches in rlmstudio.prompts.templates."""
        from rlmstudio.prompts.templates import _template_cache

        # Seed a fake cache entry
        _template_cache["__test_sentinel__"] = "stale"
        reload_system_prompt_templates()
        assert "__test_sentinel__" not in _template_cache

    def test_reload_updates_module_constants(self):
        """reload refreshes the module-level SYSTEM_PROMPT_TEMPLATES dict."""
        original_keys = set(SYSTEM_PROMPT_TEMPLATES.keys())
        reload_system_prompt_templates()
        assert set(SYSTEM_PROMPT_TEMPLATES.keys()) == original_keys
