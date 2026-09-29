"""Tests for the official-engine construction wiring in ``AppState``.

Every other test in the suite replaces ``create_rlm_engine`` /
``create_rlm_engine_for_chat_provider`` with a lambda, so the real factories
in ``server/dependencies.py`` never run.  These tests pin that seam: what
``AppState`` actually hands to :class:`RlmsEngineAdapter` for a given
LLM Provider, sandbox config and slot adapter.

The assertions are on the **constructor kwargs**, captured by swapping
``dependencies.RlmsEngineAdapter`` for a recorder.  That keeps this file
independent of the adapter's internals — how the adapter *uses* those kwargs
is covered by ``tests/test_rlms_adapter.py``.

``_get_instance_api_key`` reads the real keyring / file secret store, so an
autouse fixture stubs it to return ``None``.  No test here touches a real
credential; every key is a literal sentinel.
"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from datetime import datetime, timezone
from typing import Any

import pytest

from rlmstudio.application.dto import LLMResponseDTO
from rlmstudio.application.sandbox_vars import MODE_RLM_OFFICIAL
from rlmstudio.server import dependencies as deps
from rlmstudio.server.dependencies import AppState, get_state, reset_state
from rlmstudio.server.models import (
    ChatProviderConfig,
    LLMProviderConfig,
    ProviderConfig,
    RuntimeSettings,
)
from rlmstudio.ui.data.providers_catalog import PROVIDERS_BY_KEY

# Backend catalog keys under test — never spelled as bare literals below.
_BACKEND_OPENAI = "openai"
_BACKEND_ANTHROPIC = "anthropic"
_BACKEND_OLLAMA = "ollama"
_BACKEND_LMSTUDIO = "lmstudio"

# Sentinel credentials.  These are not real keys and are never sent anywhere.
_INSTANCE_KEY = "sk-instance-sentinel"
_ENV_KEY = "sk-env-sentinel"

_CUSTOM_ENDPOINT = "http://gpu-box:9000/v1"
_DOCKER_IMAGE = "rlm-studio-sandbox:test"
_SANDBOX_DOCKER = "docker"
_SANDBOX_RESTRICTED = "restricted"

_RLM_ENGINE_PORT_METHODS = ("run", "run_async", "is_available")


class _AdapterRecorder:
    """Stand-in for ``RlmsEngineAdapter`` that records its constructor kwargs."""

    last_kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_kwargs = kwargs
        self.kwargs = kwargs

    @property
    def version(self) -> str | None:
        return None

    def is_available(self) -> tuple[bool, str]:
        return True, ""

    def run(self, content: str, query: str, config: Any) -> Any:  # pragma: no cover - not called
        raise AssertionError("the recorder is never run")

    async def run_async(
        self, content: str, query: str, config: Any
    ) -> Any:  # pragma: no cover - not called
        raise AssertionError("the recorder is never run")


class _FakeLLMAdapter:
    """Slot LLM adapter whose ``get_completion_cost`` is the expected ``cost_fn``."""

    def complete(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
        return LLMResponseDTO(content="", model="fake", input_tokens=0, output_tokens=0)

    def count_tokens(self, text: str) -> int:
        return 1

    def get_pricing(self) -> dict[str, float]:
        return {"input_cost_per_1m": 0.0, "output_cost_per_1m": 0.0}

    def get_completion_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.5


class _CostlessLLMAdapter:
    """Slot LLM adapter with no cost table — ``cost_fn`` must come out ``None``."""

    def complete(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
        return LLMResponseDTO(content="", model="fake", input_tokens=0, output_tokens=0)

    def count_tokens(self, text: str) -> int:
        return 1

    def get_pricing(self) -> dict[str, float]:
        return {"input_cost_per_1m": 0.0, "output_cost_per_1m": 0.0}


@pytest.fixture(autouse=True)
def _clean_state() -> Generator[None, None, None]:
    reset_state()
    yield
    reset_state()


@pytest.fixture(autouse=True)
def _no_secret_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the real keyring / file secret store out of every test in this file."""
    monkeypatch.setattr(deps, "_get_instance_api_key", lambda provider_id, backend: None)


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> type[_AdapterRecorder]:
    """Swap the adapter class the factories instantiate for a kwargs recorder."""
    _AdapterRecorder.last_kwargs = {}
    monkeypatch.setattr(deps, "RlmsEngineAdapter", _AdapterRecorder)
    return _AdapterRecorder


def _llm_provider(
    *,
    backend: str = _BACKEND_OPENAI,
    model: str = "gpt-4o-mini",
    endpoint: str | None = None,
) -> LLMProviderConfig:
    return LLMProviderConfig(
        id=str(uuid.uuid4()),
        name=f"{backend}-instance",
        backend=backend,
        model=model,
        endpoint=endpoint,
    )


def _chat_provider(llm_provider_id: str = "") -> ChatProviderConfig:
    now = datetime.now(timezone.utc)
    return ChatProviderConfig(
        id=str(uuid.uuid4()),
        name="Official",
        llm_provider_id=llm_provider_id,
        execution_mode=MODE_RLM_OFFICIAL,
        runtime_settings=RuntimeSettings(),
        created_at=now,
        updated_at=now,
    )


def _register(state: AppState, lp: LLMProviderConfig) -> ChatProviderConfig:
    """Store an LLM Provider plus a Chat Provider pointing at it."""
    state.config.llm_providers.append(lp)
    cp = _chat_provider(lp.id)
    state.config.chat_providers.append(cp)
    return cp


class TestChatProviderPath:
    """Pins ``AppState.create_rlm_engine_for_chat_provider`` (dependencies.py:991)."""

    def test_backend_and_model_come_from_the_llm_provider(
        self, recorder: type[_AdapterRecorder]
    ) -> None:
        state = get_state()
        lp = _llm_provider(backend=_BACKEND_ANTHROPIC, model="claude-sonnet-4-6")
        cp = _register(state, lp)

        state.create_rlm_engine_for_chat_provider(cp.id)

        assert recorder.last_kwargs["backend"] == _BACKEND_ANTHROPIC
        assert recorder.last_kwargs["model"] == "claude-sonnet-4-6"

    def test_api_key_comes_from_the_provider_instance_key(
        self, recorder: type[_AdapterRecorder], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = get_state()
        lp = _llm_provider(backend=_BACKEND_OPENAI)
        cp = _register(state, lp)
        seen: list[tuple[str, str]] = []

        def _instance_key(provider_id: str, backend: str) -> str:
            seen.append((provider_id, backend))
            return _INSTANCE_KEY

        monkeypatch.setattr(deps, "_get_instance_api_key", _instance_key)

        state.create_rlm_engine_for_chat_provider(cp.id)

        assert seen == [(lp.id, _BACKEND_OPENAI)]
        assert recorder.last_kwargs["api_key"] == _INSTANCE_KEY

    def test_no_stored_key_yields_none_rather_than_a_placeholder(
        self, recorder: type[_AdapterRecorder]
    ) -> None:
        state = get_state()
        cp = _register(state, _llm_provider(backend=_BACKEND_OPENAI))

        state.create_rlm_engine_for_chat_provider(cp.id)

        assert recorder.last_kwargs["api_key"] is None

    def test_explicit_endpoint_is_passed_through(self, recorder: type[_AdapterRecorder]) -> None:
        state = get_state()
        lp = _llm_provider(backend=_BACKEND_LMSTUDIO, model="qwen3:8b", endpoint=_CUSTOM_ENDPOINT)
        cp = _register(state, lp)

        state.create_rlm_engine_for_chat_provider(cp.id)

        assert recorder.last_kwargs["base_url"] == _CUSTOM_ENDPOINT

    def test_base_url_falls_back_to_the_catalog_default_endpoint(
        self, recorder: type[_AdapterRecorder]
    ) -> None:
        state = get_state()
        lp = _llm_provider(backend=_BACKEND_OLLAMA, model="llama3.2", endpoint=None)
        cp = _register(state, lp)

        state.create_rlm_engine_for_chat_provider(cp.id)

        expected = PROVIDERS_BY_KEY[_BACKEND_OLLAMA].default_endpoint
        assert expected  # the catalog really does carry a default for this provider
        assert recorder.last_kwargs["base_url"] == expected

    def test_legacy_chat_provider_without_an_llm_provider_uses_its_own_fields(
        self, recorder: type[_AdapterRecorder]
    ) -> None:
        """An unmigrated Chat Provider still resolves via its deprecated fields."""
        state = get_state()
        cp = _chat_provider()
        cp.llm_provider = _BACKEND_OLLAMA
        cp.llm_model = "mistral"
        state.config.chat_providers.append(cp)
        state.config.provider_configs.append(
            ProviderConfig(provider=_BACKEND_OLLAMA, model="mistral", endpoint=_CUSTOM_ENDPOINT)
        )

        state.create_rlm_engine_for_chat_provider(cp.id)

        assert recorder.last_kwargs["backend"] == _BACKEND_OLLAMA
        assert recorder.last_kwargs["model"] == "mistral"
        assert recorder.last_kwargs["base_url"] == _CUSTOM_ENDPOINT

    def test_unknown_chat_provider_is_a_value_error(self, recorder: type[_AdapterRecorder]) -> None:
        with pytest.raises(ValueError, match="not found"):
            get_state().create_rlm_engine_for_chat_provider("no-such-provider")

        assert recorder.last_kwargs == {}


class TestActiveBackendPath:
    """Pins ``AppState.create_rlm_engine`` (dependencies.py:1007), the no-Chat-Provider path."""

    def test_backend_and_model_come_from_the_active_provider(
        self, recorder: type[_AdapterRecorder]
    ) -> None:
        state = get_state()
        state.config.active_provider = _BACKEND_OPENAI
        state.config.active_model = "gpt-4o"

        state.create_rlm_engine(None)

        assert recorder.last_kwargs["backend"] == _BACKEND_OPENAI
        assert recorder.last_kwargs["model"] == "gpt-4o"

    def test_api_key_comes_from_the_catalog_env_var(
        self, recorder: type[_AdapterRecorder], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = get_state()
        state.config.active_provider = _BACKEND_OPENAI
        state.config.active_model = "gpt-4o-mini"
        env_var = PROVIDERS_BY_KEY[_BACKEND_OPENAI].env_var
        assert env_var  # the catalog really does name an env var for this provider
        monkeypatch.setenv(env_var, _ENV_KEY)

        state.create_rlm_engine(None)

        assert recorder.last_kwargs["api_key"] == _ENV_KEY

    def test_provider_without_an_env_var_gets_no_key(
        self, recorder: type[_AdapterRecorder]
    ) -> None:
        state = get_state()
        state.config.active_provider = _BACKEND_OLLAMA
        state.config.active_model = "llama3.2"
        assert PROVIDERS_BY_KEY[_BACKEND_OLLAMA].env_var is None

        state.create_rlm_engine(None)

        assert recorder.last_kwargs["api_key"] is None

    def test_base_url_falls_back_to_the_catalog_default_endpoint(
        self, recorder: type[_AdapterRecorder]
    ) -> None:
        state = get_state()
        state.config.active_provider = _BACKEND_LMSTUDIO
        state.config.active_model = "qwen3:8b"

        state.create_rlm_engine(None)

        expected = PROVIDERS_BY_KEY[_BACKEND_LMSTUDIO].default_endpoint
        assert expected
        assert recorder.last_kwargs["base_url"] == expected

    def test_enabled_provider_config_endpoint_wins_over_the_catalog_default(
        self, recorder: type[_AdapterRecorder]
    ) -> None:
        state = get_state()
        state.config.active_provider = _BACKEND_LMSTUDIO
        state.config.active_model = "qwen3:8b"
        state.config.provider_configs.append(
            ProviderConfig(
                provider=_BACKEND_LMSTUDIO,
                model="qwen3:8b",
                endpoint=_CUSTOM_ENDPOINT,
                enabled=True,
            )
        )

        state.create_rlm_engine(None)

        assert recorder.last_kwargs["base_url"] == _CUSTOM_ENDPOINT


class TestSandboxAndCostWiring:
    """Pins ``AppState._build_rlm_engine`` (dependencies.py:1016) for both entry points."""

    def test_docker_sandbox_and_image_come_from_the_config(
        self, recorder: type[_AdapterRecorder]
    ) -> None:
        state = get_state()
        state.config.sandbox.type = _SANDBOX_DOCKER
        state.config.sandbox.docker_image = _DOCKER_IMAGE
        cp = _register(state, _llm_provider())

        state.create_rlm_engine_for_chat_provider(cp.id)

        assert recorder.last_kwargs["sandbox_type"] == _SANDBOX_DOCKER
        assert recorder.last_kwargs["docker_image"] == _DOCKER_IMAGE

    def test_non_docker_sandbox_type_is_passed_through_verbatim(
        self, recorder: type[_AdapterRecorder]
    ) -> None:
        state = get_state()
        state.config.sandbox.type = _SANDBOX_RESTRICTED
        state.config.sandbox.docker_image = None

        state.create_rlm_engine(None)

        assert recorder.last_kwargs["sandbox_type"] == _SANDBOX_RESTRICTED
        assert recorder.last_kwargs["docker_image"] is None

    def test_cost_fn_is_the_slot_adapters_cost_table(
        self, recorder: type[_AdapterRecorder]
    ) -> None:
        state = get_state()
        cp = _register(state, _llm_provider())
        llm = _FakeLLMAdapter()

        state.create_rlm_engine_for_chat_provider(cp.id, llm)

        cost_fn = recorder.last_kwargs["cost_fn"]
        assert cost_fn == llm.get_completion_cost
        assert cost_fn(1_000, 100) == llm.get_completion_cost(1_000, 100)

    def test_cost_fn_is_none_without_a_slot_adapter(self, recorder: type[_AdapterRecorder]) -> None:
        get_state().create_rlm_engine(None)

        assert recorder.last_kwargs["cost_fn"] is None

    def test_cost_fn_is_none_when_the_adapter_has_no_cost_table(
        self, recorder: type[_AdapterRecorder]
    ) -> None:
        state = get_state()
        cp = _register(state, _llm_provider())

        state.create_rlm_engine_for_chat_provider(cp.id, _CostlessLLMAdapter())

        assert recorder.last_kwargs["cost_fn"] is None


class TestFactoriesReturnAnEnginePort:
    """Both factories return something usable as an ``RLMEnginePort`` — no recorder."""

    def test_chat_provider_factory_returns_an_engine_port(self) -> None:
        state = get_state()
        cp = _register(state, _llm_provider())

        engine = state.create_rlm_engine_for_chat_provider(cp.id, _FakeLLMAdapter())

        assert all(callable(getattr(engine, name, None)) for name in _RLM_ENGINE_PORT_METHODS)
        assert engine.version is None or isinstance(engine.version, str)

    def test_active_backend_factory_returns_an_engine_port(self) -> None:
        engine = get_state().create_rlm_engine(_FakeLLMAdapter())

        assert all(callable(getattr(engine, name, None)) for name in _RLM_ENGINE_PORT_METHODS)
        available, reason = engine.is_available()
        assert isinstance(available, bool)
        assert reason  # never silent: either the version or the install hint
