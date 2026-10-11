# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0
"""
LLMConfigManager - Secure management of LLM provider configurations.
"""

import json
import os
from pathlib import Path
from typing import Any

from rlmstudio.branding import state_dir

from .models import LLMProviderConfig


def load_env_file(env_path: Path | None = None) -> None:
    """
    Load environment variables from .env file.
    """
    if env_path is None:
        env_path = Path(".env")

    if not env_path.exists():
        return

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key] = value


def update_env_file(key: str, value: str, env_path: Path | None = None) -> None:
    """
    Add or update a key-value pair in the .env file.
    """
    if env_path is None:
        env_path = Path(".env")
    lines = []
    if env_path.exists():
        with open(env_path) as f:
            lines = f.readlines()
    found = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[i] = f"{key}={value}\n"
            found = True
            break
    if not found:
        lines.append(f"{key}={value}\n")
    with open(env_path, "w") as f:
        f.writelines(lines)


class LLMConfigManager:
    """
    Manage LLM provider configurations securely.
    ...
    """

    def __init__(self, config_dir: Path | None = None):
        if config_dir is None:
            config_dir = state_dir()
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True, mode=0o700)
        self._configs: dict[str, LLMProviderConfig] = {}
        self._active_provider: str | None = None
        self._load_all_configs()

    def _load_config(self, provider: str) -> None:
        """
        Load configuration from disk and load .env file to ensure API keys are available.
        """
        # Load .env file to ensure environment variables are available
        load_env_file()

        config_file = self.config_dir / f"{provider}.json"
        if not config_file.exists():
            return
        with open(config_file) as f:
            config_dict = json.load(f)
        api_key_env_var = config_dict.get("api_key_env_var")
        if api_key_env_var:
            config_dict["api_key"] = os.getenv(api_key_env_var)
        # Filter out unknown keys for forward-compat (e.g., old files with extra fields)
        known_fields = set(LLMProviderConfig.__dataclass_fields__.keys())
        filtered_dict = {k: v for k, v in config_dict.items() if k in known_fields}
        config = LLMProviderConfig(**filtered_dict)
        self._configs[provider] = config

    def _load_all_configs(self) -> None:
        """Load all provider configurations from disk."""
        for config_file in self.config_dir.glob("*.json"):
            provider = config_file.stem
            # Skip non-provider files
            if provider not in ["encryption", "api_keys", "system_prompt_custom", "profiles"]:
                self._load_config(provider)

    def _save_config(self, config: LLMProviderConfig) -> None:
        """
        Save configuration to disk (without API key).
        """
        config_file = self.config_dir / f"{config.provider}.json"
        config_dict = config.to_dict(include_api_key=False)
        with open(config_file, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
        try:
            config_file.chmod(0o600)
        except Exception:
            pass

    def list_providers(self) -> list:
        """Get list of configured providers."""
        return list(self._configs.keys())

    def get_provider_config(self, provider: str) -> LLMProviderConfig | None:
        """
        Get configuration for a provider.
        """
        if provider not in self._configs:
            self._load_config(provider)
        return self._configs.get(provider)

    def get_active_provider(self) -> LLMProviderConfig | None:
        """
        Get currently active provider configuration.
        """
        if self._active_provider:
            return self.get_provider_config(self._active_provider)
        return None

    # Known valid models per provider (for offline validation)
    VALID_MODELS = {
        "openai": {"gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo", "gpt-4o-mini"},
        "anthropic": {
            "claude-3-opus",
            "claude-3-sonnet",
            "claude-3-haiku",
            "claude-3-5-sonnet",
            "claude-3-5-haiku",
        },
    }

    def test_connection(
        self,
        provider: str,
        model: str,
        api_key: str | None = None,
        api_key_env_var: str | None = None,
    ) -> bool:
        """
        Test connection to a provider.

        Returns:
            bool: True if connection parameters are valid, False otherwise.
        """
        # Basic validation
        if not provider or not model:
            return False

        # Resolve API key from env var if provided
        resolved_key = api_key
        if not resolved_key and api_key_env_var:
            resolved_key = os.getenv(api_key_env_var)

        # Provider-specific validation
        if provider == "openai":
            # OpenAI keys must start with sk-
            if not resolved_key or not resolved_key.startswith("sk-"):
                return False
            # Validate model name
            if model not in self.VALID_MODELS.get("openai", set()):
                return False
            return True

        elif provider == "anthropic":
            # Anthropic keys must start with sk-ant-
            if not resolved_key or not resolved_key.startswith("sk-ant-"):
                return False
            # Validate model name
            if model not in self.VALID_MODELS.get("anthropic", set()):
                return False
            return True

        elif provider in ("ollama", "lmstudio"):
            # Local providers don't need API key, just a model name
            return True

        else:
            # Unknown provider
            return False

    def add_provider(
        self,
        provider: str,
        model: str,
        api_key: str | None = None,
        api_key_env_var: str | None = None,
        input_cost_per_1k: float = 0.0,
        output_cost_per_1k: float = 0.0,
        **kwargs: Any,
    ) -> tuple:
        """
        Add or update a provider configuration.
        ...
        """
        if not api_key and not api_key_env_var:
            return False, "Must provide either api_key or api_key_env_var"

        config = LLMProviderConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            api_key_env_var=api_key_env_var,
            input_cost_per_1k_tokens=input_cost_per_1k,
            output_cost_per_1k_tokens=output_cost_per_1k,
            **kwargs,
        )

        # Test connection before saving
        if not self.test_connection(provider, model, api_key, api_key_env_var):
            print(f"DEBUG: test_connection returned False for {provider}")
            return False, "Connection test failed"

        # Mark test as successful
        config.test_successful = True
        print(f"DEBUG: Set test_successful=True for {provider}")

        # Record the env var name so config knows where to look on reload
        if api_key:
            from rlmstudio.ui.data.providers_catalog import get_env_var

            env_var = get_env_var(provider) or f"{provider.upper()}_API_KEY"
            config.api_key_env_var = env_var
            config.api_key = None  # Do not keep in memory

        # Save configuration
        self._save_config(config)
        self._configs[provider] = config
        print(
            f"DEBUG: Saved config. is_ready={config.is_ready}, test_successful={config.test_successful}"
        )

        # Set as active provider (auto-select first provider added)
        if not self._active_provider:
            self._active_provider = provider

        return True, ""

    def delete_provider(self, provider: str) -> bool:
        """
        Delete a provider configuration file.

        Args:
            provider: Provider name to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            # Path to provider config file
            config_file = self.config_dir / f"{provider}.json"

            # Delete the file if it exists
            if config_file.exists():
                config_file.unlink()

            # Remove from loaded configs
            if provider in self._configs:
                del self._configs[provider]

            # If this was the active provider, clear it
            if self._active_provider == provider:
                self._active_provider = None

            return True
        except Exception as e:
            print(f"DEBUG: Failed to delete provider {provider}: {e}")
            return False
