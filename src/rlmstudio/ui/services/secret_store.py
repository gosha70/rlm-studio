# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0
"""
SecretStore — Centralized API key persistence with explicit storage policy.

Three implementations:
  - EnvSecretStore:     reads from environment variables only (never writes)
  - FileSecretStore:    reads/writes <state dir>/api_keys.json  (chmod 600)
  - KeyringSecretStore: uses the `keyring` library (optional dependency)
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast

from rlmstudio.branding import KEYRING_SERVICE, LEGACY_KEYRING_SERVICES, state_dir
from rlmstudio.ui.data.providers_catalog import get_env_var

# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class SecretStore(ABC):
    """Interface for API-key storage backends."""

    @abstractmethod
    def get(self, provider: str) -> str | None:
        """Return the API key for *provider*, or ``None``."""

    @abstractmethod
    def set(self, provider: str, key: str) -> None:
        """Persist *key* for *provider*."""

    @abstractmethod
    def delete(self, provider: str) -> None:
        """Remove any stored key for *provider*."""

    # Convenience ---------------------------------------------------------

    @property
    def label(self) -> str:
        """Human-readable name shown in the UI."""
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# 1. Environment variables only (read-only, recommended)
# ---------------------------------------------------------------------------


class EnvSecretStore(SecretStore):
    """Read API keys from environment variables.  Never writes anything."""

    label = "Environment variables only"

    def get(self, provider: str) -> str | None:
        env_var = get_env_var(provider)
        if env_var:
            return os.environ.get(env_var)
        return None

    def set(self, provider: str, key: str) -> None:
        # Env-only store: put into the process env so the current session
        # can use it, but nothing is persisted to disk.
        env_var = get_env_var(provider)
        if env_var:
            os.environ[env_var] = key

    def delete(self, provider: str) -> None:
        env_var = get_env_var(provider)
        if env_var:
            os.environ.pop(env_var, None)


# ---------------------------------------------------------------------------
# 2. Local JSON file  (<state dir>/api_keys.json, chmod 600)
# ---------------------------------------------------------------------------

_DEFAULT_KEYS_PATH = state_dir() / "api_keys.json"


class FileSecretStore(SecretStore):
    """Read/write API keys to a local JSON file with restricted permissions."""

    label = "Save to local file (unencrypted)"

    def __init__(self, path: Path | None = None):
        self._path = Path(path) if path else _DEFAULT_KEYS_PATH

    # -- helpers ----------------------------------------------------------

    def _read_all(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        try:
            with open(self._path) as f:
                return cast(dict[str, Any], json.load(f))
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_all(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2)
        try:
            self._path.chmod(0o600)
        except OSError:
            pass  # Windows

    # -- interface --------------------------------------------------------

    def get(self, provider: str) -> str | None:
        return self._read_all().get(provider)

    def set(self, provider: str, key: str) -> None:
        data = self._read_all()
        data[provider] = key
        self._write_all(data)
        # Also inject into process env so current session can use it
        env_var = get_env_var(provider)
        if env_var:
            os.environ[env_var] = key

    def delete(self, provider: str) -> None:
        data = self._read_all()
        if provider in data:
            del data[provider]
            self._write_all(data)


# ---------------------------------------------------------------------------
# 3. System keyring (optional dependency)
# ---------------------------------------------------------------------------

_KEYRING_SERVICE = KEYRING_SERVICE


class KeyringSecretStore(SecretStore):
    """Use the OS keyring (Keychain / Windows Credential Locker / …)."""

    label = "Save to system keyring"

    @staticmethod
    def is_available() -> bool:
        """Return True if the ``keyring`` package is importable."""
        try:
            import keyring  # noqa: F401

            return True
        except ImportError:
            return False

    def get(self, provider: str) -> str | None:
        import keyring

        result = keyring.get_password(_KEYRING_SERVICE, provider)
        if result is not None:
            return str(result)
        # Fall back to keys saved under an older service name and migrate
        # them forward so the next read hits the canonical service.
        for legacy_service in LEGACY_KEYRING_SERVICES:
            legacy = keyring.get_password(legacy_service, provider)
            if legacy is not None:
                keyring.set_password(_KEYRING_SERVICE, provider, str(legacy))
                return str(legacy)
        return None

    def set(self, provider: str, key: str) -> None:
        import keyring

        keyring.set_password(_KEYRING_SERVICE, provider, key)
        # Also inject into process env so current session can use it
        env_var = get_env_var(provider)
        if env_var:
            os.environ[env_var] = key

    def delete(self, provider: str) -> None:
        import keyring

        try:
            keyring.delete_password(_KEYRING_SERVICE, provider)
        except keyring.errors.PasswordDeleteError:
            pass


# ---------------------------------------------------------------------------
# Factory / resolution helpers
# ---------------------------------------------------------------------------

# Ordered list of policy choices shown in the UI
KEY_POLICY_OPTIONS: list[tuple[str, str]] = [
    ("env", "Environment variables only (recommended)"),
    ("file", "Save to local file (unencrypted)"),
]
# Conditionally add keyring option
if KeyringSecretStore.is_available():
    KEY_POLICY_OPTIONS.append(("keyring", "Save to system keyring"))


def create_secret_store(policy: str, **kwargs: Any) -> SecretStore:
    """Instantiate the right SecretStore for a policy key.

    Args:
        policy: One of ``"env"``, ``"file"``, ``"keyring"``.
    """
    if policy == "env":
        return EnvSecretStore()
    if policy == "file":
        return FileSecretStore(**kwargs)
    if policy == "keyring":
        if not KeyringSecretStore.is_available():
            raise RuntimeError(
                "The 'keyring' package is not installed. Install it with: pip install keyring"
            )
        return KeyringSecretStore()
    raise ValueError(f"Unknown key policy: {policy!r}")


def get_secret_store(session_state: object = None) -> SecretStore:
    """Return the active SecretStore from session state, or a sensible default.

    The config panel stores ``st.session_state.key_policy`` (str) which is
    one of ``"env"``, ``"file"``, ``"keyring"``.
    """
    policy = "file"  # default: backward-compatible with existing behaviour
    if session_state is not None:
        policy = getattr(session_state, "key_policy", None) or "file"
    return create_secret_store(policy)


def resolve_api_key(provider: str, session_state: object = None) -> str | None:
    """Try every available source to find an API key for *provider*.

    Resolution order:
      1. Session-state in-memory cache  (``provider_api_keys``)
      2. Active SecretStore (per user's key_policy)
      3. Environment variable fallback
    """
    # 1. In-memory session cache
    if session_state is not None:
        cache = getattr(session_state, "provider_api_keys", None) or {}
        if provider in cache:
            return str(cache[provider])

    # 2. Active store
    store = get_secret_store(session_state)
    key = store.get(provider)
    if key:
        return key

    # 3. Env fallback (covers cases where user set the var externally)
    env_var = get_env_var(provider)
    if env_var:
        return os.environ.get(env_var)

    return None
