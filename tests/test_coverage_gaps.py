"""Targeted tests to close coverage gaps in production modules.

Covers:
- infrastructure/config (yaml_config, __init__)
- infrastructure/storage (sqlite_adapter, __init__)
- infrastructure/llm adapters (anthropic, openai, ollama)
- infrastructure/sandbox adapters (execute_async on all three)
- server/judge (_parse_json_from_response, JudgeService sync paths)
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Tiny branch gaps — each covers 1-2 missing statements
# ---------------------------------------------------------------------------


class TestTinyBranchGaps:
    def test_dataset_load_non_dict_case_raises(self, tmp_path: Path) -> None:
        """benchmark/dataset.py line 136 — non-dict case entry."""
        import json

        from rlmstudio.benchmark.dataset import BenchmarkDataset

        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"cases": ["not-a-dict"]}))
        with pytest.raises((ValueError, Exception)):
            BenchmarkDataset.load(str(bad))

    def test_benchmark_report_csv_empty(self) -> None:
        """benchmark/report.py line 164 — empty case_results → to_csv_string returns ''."""
        from rlmstudio.benchmark.report import BenchmarkReport
        from rlmstudio.benchmark.runner import BenchmarkRun

        run = BenchmarkRun(dataset_name="test", strategy_names=["direct"])
        report = BenchmarkReport(run)
        assert report.to_csv_string() == ""

    def test_budget_tracker_elapsed_none_when_not_started(self) -> None:
        """core/budget.py line 204 — elapsed_time returns None before start() called."""
        from rlmstudio.core.budget import BudgetTracker

        tracker = BudgetTracker()
        assert tracker.elapsed_time is None

    def test_provider_id_eq_str(self) -> None:
        """domain/value_objects.py line 119 — ProviderId.__eq__ with str."""
        from rlmstudio.domain.value_objects import ProviderId

        pid = ProviderId("openai")
        assert pid == "openai"
        assert pid != "anthropic"
        assert pid.__eq__(42) is NotImplemented

    def test_provider_id_eq_other_type_returns_not_implemented(self) -> None:
        """domain/value_objects.py — ProviderId.__eq__ with unrelated type."""
        from rlmstudio.domain.value_objects import ProviderId

        pid = ProviderId("openai")
        assert pid.__eq__(3.14) is NotImplemented

    def test_database_default_path(self) -> None:
        """storage/database.py line 93 — Database() uses get_default_db_path()."""
        from rlmstudio.storage.database import Database, get_default_db_path

        db = Database()  # triggers get_default_db_path()
        assert db.db_path == get_default_db_path()
        db.close()


# ---------------------------------------------------------------------------
# ui/utils/__init__.py — pure formatting logic, no Streamlit dependency
# ---------------------------------------------------------------------------


class TestUIUtils:
    def test_format_tokens_small(self) -> None:
        from rlmstudio.ui.utils import format_tokens

        assert format_tokens(500) == "500"

    def test_format_tokens_thousands(self) -> None:
        from rlmstudio.ui.utils import format_tokens

        assert format_tokens(2500) == "2.5K"

    def test_format_tokens_millions(self) -> None:
        from rlmstudio.ui.utils import format_tokens

        assert format_tokens(1_500_000) == "1.5M"

    def test_format_cost(self) -> None:
        from rlmstudio.ui.utils import format_cost

        assert format_cost(0.0123) == "$0.0123"

    def test_format_time_minutes(self) -> None:
        from rlmstudio.ui.utils import format_time

        assert format_time(90) == "1.5m"

    def test_format_time_seconds(self) -> None:
        from rlmstudio.ui.utils import format_time

        assert format_time(3.5) == "3.5s"

    def test_format_time_milliseconds(self) -> None:
        from rlmstudio.ui.utils import format_time

        assert format_time(0.123) == "123ms"

    def test_format_memory_gb(self) -> None:
        from rlmstudio.ui.utils import format_memory

        assert format_memory(2048.0) == "2.0 GB"

    def test_format_memory_mb(self) -> None:
        from rlmstudio.ui.utils import format_memory

        assert format_memory(512.0) == "512.0 MB"

    def test_format_memory_kb(self) -> None:
        from rlmstudio.ui.utils import format_memory

        assert "KB" in format_memory(0.5)

    def test_format_memory_bytes(self) -> None:
        from rlmstudio.ui.utils import format_memory

        assert "B" in format_memory(0.0000001)

    def test_format_memory_zero(self) -> None:
        from rlmstudio.ui.utils import format_memory

        assert format_memory(0.0) == "0 KB"

    def test_format_percentage_positive(self) -> None:
        from rlmstudio.ui.utils import format_percentage

        assert format_percentage(5.5) == "+5.5%"

    def test_format_percentage_negative(self) -> None:
        from rlmstudio.ui.utils import format_percentage

        assert format_percentage(-3.2) == "-3.2%"

    def test_format_percentage_savings(self) -> None:
        from rlmstudio.ui.utils import format_percentage_savings

        result = format_percentage_savings(100.0, 75.0)
        assert "25.0" in result

    def test_format_percentage_savings_zero_original(self) -> None:
        from rlmstudio.ui.utils import format_percentage_savings

        assert format_percentage_savings(0.0, 50.0) == "N/A"

    def test_constants_accessible(self) -> None:
        from rlmstudio.ui.utils import APP_TITLE, COLORS, EXECUTION_MODES, PROVIDERS

        assert APP_TITLE == "RLMKit Chat Studio"
        assert "rlm" in COLORS
        assert "compare" in EXECUTION_MODES
        assert "openai" in PROVIDERS


# ---------------------------------------------------------------------------
# infrastructure/config
# ---------------------------------------------------------------------------


class TestInfrastructureConfig:
    def test_import(self) -> None:
        from rlmstudio.infrastructure.config import load_config

        assert callable(load_config)

    def test_load_config_defaults(self) -> None:
        from rlmstudio.infrastructure.config import load_config

        cfg = load_config()
        assert cfg.max_steps is not None
        assert cfg.max_time_seconds is not None

    def test_load_config_returns_run_config_dto(self) -> None:
        from rlmstudio.application.dto import RunConfigDTO
        from rlmstudio.infrastructure.config import load_config

        cfg = load_config()
        assert isinstance(cfg, RunConfigDTO)
        assert "safe_mode" in cfg.extra


# ---------------------------------------------------------------------------
# infrastructure/storage
# ---------------------------------------------------------------------------


class TestSQLiteStorageAdapter:
    def test_import(self) -> None:
        from rlmstudio.infrastructure.storage import SQLiteStorageAdapter

        assert SQLiteStorageAdapter is not None

    def test_create_and_get_conversation(self, tmp_path: Path) -> None:
        from rlmstudio.infrastructure.storage import SQLiteStorageAdapter

        adapter = SQLiteStorageAdapter(str(tmp_path / "test.db"))
        cid = adapter.create_conversation("My Chat", "direct", "openai", "gpt-4")
        assert isinstance(cid, str)
        assert len(cid) > 0

        conv = adapter.get_conversation(cid)
        assert conv is not None
        adapter.close()

    def test_get_nonexistent_conversation_returns_none(self, tmp_path: Path) -> None:
        from rlmstudio.infrastructure.storage import SQLiteStorageAdapter

        adapter = SQLiteStorageAdapter(str(tmp_path / "test.db"))
        assert adapter.get_conversation("does-not-exist") is None
        adapter.close()

    def test_list_conversations(self, tmp_path: Path) -> None:
        from rlmstudio.infrastructure.storage import SQLiteStorageAdapter

        adapter = SQLiteStorageAdapter(str(tmp_path / "test.db"))
        adapter.create_conversation("A")
        adapter.create_conversation("B")
        convs = adapter.list_conversations()
        assert len(convs) == 2
        adapter.close()

    def test_delete_conversation(self, tmp_path: Path) -> None:
        from rlmstudio.infrastructure.storage import SQLiteStorageAdapter

        adapter = SQLiteStorageAdapter(str(tmp_path / "test.db"))
        cid = adapter.create_conversation()
        adapter.delete_conversation(cid)
        assert adapter.get_conversation(cid) is None
        adapter.close()

    def test_file_context_roundtrip(self, tmp_path: Path) -> None:
        from rlmstudio.infrastructure.storage import SQLiteStorageAdapter

        adapter = SQLiteStorageAdapter(str(tmp_path / "test.db"))
        content = "Hello, world! " * 100
        hash_id = adapter.save_file_context(content, "doc.txt")
        assert isinstance(hash_id, str)
        retrieved = adapter.get_file_context(hash_id)
        assert retrieved == content
        adapter.close()

    def test_get_file_context_missing_returns_none(self, tmp_path: Path) -> None:
        from rlmstudio.infrastructure.storage import SQLiteStorageAdapter

        adapter = SQLiteStorageAdapter(str(tmp_path / "test.db"))
        assert adapter.get_file_context("nonexistent-hash") is None
        adapter.close()

    def test_add_and_search_chunks(self, tmp_path: Path) -> None:
        from rlmstudio.infrastructure.storage import SQLiteStorageAdapter

        adapter = SQLiteStorageAdapter(str(tmp_path / "test.db"))
        n = adapter.add_chunks(
            "col1",
            ["chunk A", "chunk B"],
            [[1.0, 0.0], [0.0, 1.0]],
            source_id="src1",
        )
        assert n >= 0
        results = adapter.search_chunks("col1", [1.0, 0.0], top_k=2)
        assert isinstance(results, list)
        adapter.close()


# ---------------------------------------------------------------------------
# infrastructure/llm adapters
# ---------------------------------------------------------------------------


class _FakeMetaClient:
    """Mock client that supports complete_with_metadata."""

    model = "gpt-4"

    class _Resp:
        content = "hello"
        model = "gpt-4"
        input_tokens = 5
        output_tokens = 3
        finish_reason = "stop"

    def complete_with_metadata(self, messages: list[dict[str, str]]) -> _FakeMetaClient._Resp:
        return self._Resp()

    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4


class _FakePlainClient:
    """Mock client that only supports plain complete()."""

    model = "gpt-4"

    def complete(self, messages: list[dict[str, str]]) -> str:
        return "plain response"


class TestOpenAIAdapter:
    def test_complete_with_metadata(self) -> None:
        from rlmstudio.infrastructure.llm.openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter(_FakeMetaClient())
        resp = adapter.complete([{"role": "user", "content": "hi"}])
        assert resp.content == "hello"
        assert resp.input_tokens == 5

    def test_complete_plain_fallback(self) -> None:
        from rlmstudio.infrastructure.llm.openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter(_FakePlainClient())
        resp = adapter.complete([{"role": "user", "content": "hi"}])
        assert resp.content == "plain response"

    def test_complete_stream(self) -> None:
        from rlmstudio.infrastructure.llm.openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter(_FakeMetaClient())
        tokens = list(adapter.complete_stream([{"role": "user", "content": "hi"}]))
        assert tokens == ["hello"]

    def test_count_tokens_with_client(self) -> None:
        from rlmstudio.infrastructure.llm.openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter(_FakeMetaClient())
        assert adapter.count_tokens("hello world") > 0

    def test_count_tokens_fallback(self) -> None:
        from rlmstudio.infrastructure.llm.openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter(_FakePlainClient())
        assert adapter.count_tokens("hello world") >= 1

    def test_get_pricing_known_model(self) -> None:
        from rlmstudio.infrastructure.llm.openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter(_FakeMetaClient())
        pricing = adapter.get_pricing()
        assert "input_cost_per_1m" in pricing

    def test_get_pricing_unknown_model_falls_back(self) -> None:
        from rlmstudio.infrastructure.llm.openai_adapter import OpenAIAdapter

        client = _FakePlainClient()
        client.model = "unknown-model"  # type: ignore[assignment]
        adapter = OpenAIAdapter(client)
        pricing = adapter.get_pricing()
        assert pricing["input_cost_per_1m"] == 30.0  # gpt-4 default


class TestAnthropicAdapter:
    def test_complete_with_metadata(self) -> None:
        from rlmstudio.infrastructure.llm.anthropic_adapter import AnthropicAdapter

        adapter = AnthropicAdapter(_FakeMetaClient())
        resp = adapter.complete([{"role": "user", "content": "hi"}])
        assert resp.content == "hello"

    def test_complete_plain_fallback(self) -> None:
        from rlmstudio.infrastructure.llm.anthropic_adapter import AnthropicAdapter

        adapter = AnthropicAdapter(_FakePlainClient())
        resp = adapter.complete([{"role": "user", "content": "hi"}])
        assert resp.content == "plain response"

    def test_complete_stream(self) -> None:
        from rlmstudio.infrastructure.llm.anthropic_adapter import AnthropicAdapter

        adapter = AnthropicAdapter(_FakeMetaClient())
        tokens = list(adapter.complete_stream([{"role": "user", "content": "hi"}]))
        assert tokens == ["hello"]

    def test_count_tokens(self) -> None:
        from rlmstudio.infrastructure.llm.anthropic_adapter import AnthropicAdapter

        adapter = AnthropicAdapter(_FakePlainClient())
        assert adapter.count_tokens("hello world") >= 1

    def test_get_pricing_known(self) -> None:
        """Priced from the catalog. This previously asserted
        claude-3-opus-20240229 at $15/MTok — a retired model, so it pinned a
        price for something nobody can call."""
        from rlmstudio.domain.model_catalog import DEFAULT_ANTHROPIC_MODEL, price_per_1k
        from rlmstudio.infrastructure.llm.anthropic_adapter import AnthropicAdapter

        class _ClaudeClient:
            model = DEFAULT_ANTHROPIC_MODEL

        adapter = AnthropicAdapter(_ClaudeClient())
        pricing = adapter.get_pricing()

        rates = price_per_1k(DEFAULT_ANTHROPIC_MODEL)
        assert rates is not None
        assert pricing["input_cost_per_1m"] == rates[0] * 1000
        assert pricing["output_cost_per_1m"] == rates[1] * 1000

    def test_get_pricing_unknown_reports_zero_not_another_model_rate(self) -> None:
        """An unknown model must not be priced from a substitute.

        This asserted only that a key was present, which the old Opus-3
        fallback satisfied while overcharging the current default fivefold.
        Zero means "not priced"; a plausible wrong number means nothing is
        ever noticed."""
        from rlmstudio.infrastructure.llm.anthropic_adapter import AnthropicAdapter

        class _UnknownModelClient:
            # Deliberately not _FakePlainClient: its model is "gpt-4", which
            # the catalog knows, so it never exercised the unknown branch.
            model = "claude-something-nobody-has-shipped"

        adapter = AnthropicAdapter(_UnknownModelClient())
        pricing = adapter.get_pricing()

        assert pricing["input_cost_per_1m"] == 0.0
        assert pricing["output_cost_per_1m"] == 0.0


class TestOllamaAdapter:
    def test_complete_with_metadata(self) -> None:
        from rlmstudio.infrastructure.llm.ollama_adapter import OllamaAdapter

        adapter = OllamaAdapter(_FakeMetaClient())
        resp = adapter.complete([{"role": "user", "content": "hi"}])
        assert resp.content == "hello"

    def test_complete_plain_fallback(self) -> None:
        from rlmstudio.infrastructure.llm.ollama_adapter import OllamaAdapter

        adapter = OllamaAdapter(_FakePlainClient())
        resp = adapter.complete([{"role": "user", "content": "hi"}])
        assert resp.content == "plain response"

    def test_complete_stream(self) -> None:
        from rlmstudio.infrastructure.llm.ollama_adapter import OllamaAdapter

        adapter = OllamaAdapter(_FakeMetaClient())
        tokens = list(adapter.complete_stream([{"role": "user", "content": "hi"}]))
        assert tokens == ["hello"]

    def test_count_tokens(self) -> None:
        from rlmstudio.infrastructure.llm.ollama_adapter import OllamaAdapter

        adapter = OllamaAdapter(_FakePlainClient())
        assert adapter.count_tokens("hello") >= 1

    def test_get_pricing(self) -> None:
        from rlmstudio.infrastructure.llm.ollama_adapter import OllamaAdapter

        adapter = OllamaAdapter(_FakePlainClient())
        pricing = adapter.get_pricing()
        assert pricing["input_cost_per_1m"] == 0.0


# ---------------------------------------------------------------------------
# server/judge
# ---------------------------------------------------------------------------


class TestParseJsonFromResponse:
    def test_plain_json(self) -> None:
        from rlmstudio.server.judge import _parse_json_from_response

        result = _parse_json_from_response('{"score": 4.5}')
        assert result == {"score": 4.5}

    def test_markdown_fenced_json(self) -> None:
        from rlmstudio.server.judge import _parse_json_from_response

        text = '```json\n{"winner": "a"}\n```'
        result = _parse_json_from_response(text)
        assert result == {"winner": "a"}

    def test_markdown_no_language(self) -> None:
        from rlmstudio.server.judge import _parse_json_from_response

        text = '```\n{"key": "value"}\n```'
        result = _parse_json_from_response(text)
        assert result == {"key": "value"}

    def test_whitespace_stripped(self) -> None:
        from rlmstudio.server.judge import _parse_json_from_response

        result = _parse_json_from_response('  {"x": 1}  ')
        assert result == {"x": 1}


class TestSandboxExecuteAsync:
    """Verify execute_async delegates to execute() on all three sandbox adapters."""

    @pytest.mark.asyncio
    async def test_local_sandbox_execute_async(self) -> None:
        from rlmstudio.infrastructure.sandbox.local_sandbox import LocalSandboxAdapter

        adapter = LocalSandboxAdapter()
        result = await adapter.execute_async("x = 1 + 1")
        assert result.exception is None

    @pytest.mark.asyncio
    async def test_restricted_sandbox_execute_async(self) -> None:
        from rlmstudio.infrastructure.sandbox.restricted_sandbox import RestrictedSandboxAdapter

        adapter = RestrictedSandboxAdapter()
        result = await adapter.execute_async("x = 2 + 2")
        assert result.exception is None

    @pytest.mark.asyncio
    async def test_docker_sandbox_execute_async_delegates_to_execute(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """execute_async must call execute(); patch execute() to avoid Docker dependency."""
        from rlmstudio.application.dto import ExecutionResultDTO
        from rlmstudio.infrastructure.sandbox.docker_sandbox_adapter import DockerSandboxAdapter

        adapter = DockerSandboxAdapter.__new__(DockerSandboxAdapter)
        adapter._namespace = {}  # type: ignore[attr-defined]

        captured: list[str] = []

        def _fake_execute(code: str) -> ExecutionResultDTO:
            captured.append(code)
            return ExecutionResultDTO(stdout="ok")

        monkeypatch.setattr(adapter, "execute", _fake_execute)

        result = await adapter.execute_async("print('hello')")
        assert result.stdout == "ok"
        assert captured == ["print('hello')"]


class TestJudgeService:
    def _make_state(
        self, judge_id: str | None = None, execution_ctx: tuple | None = None
    ) -> object:
        class _Config:
            judge_chat_provider_id = judge_id

        class _State:
            config = _Config()

            def resolve_execution_context(self, execution_id: str) -> tuple | None:
                return execution_ctx

        return _State()

    def test_no_judge_provider_raises(self) -> None:
        from rlmstudio.server.judge import JudgeService

        svc = JudgeService(self._make_state(judge_id=None))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="No judge"):
            svc._get_judge_provider_id()

    def test_judge_provider_id_returned(self) -> None:
        from rlmstudio.server.judge import JudgeService

        svc = JudgeService(self._make_state(judge_id="judge-cp-1"))  # type: ignore[arg-type]
        assert svc._get_judge_provider_id() == "judge-cp-1"

    def test_execution_not_found_raises(self) -> None:
        from rlmstudio.server.judge import JudgeService

        svc = JudgeService(self._make_state(judge_id="j1", execution_ctx=None))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="not found"):
            svc._get_execution_context("missing-id")

    def test_execution_context_returned(self) -> None:
        from rlmstudio.server.judge import JudgeService

        ctx = ("sess-1", "What is X?", "X is Y.", "cp-abc")
        svc = JudgeService(self._make_state(judge_id="j1", execution_ctx=ctx))  # type: ignore[arg-type]
        session_id, query, response, cp_id = svc._get_execution_context("exec-1")
        assert session_id == "sess-1"
        assert query == "What is X?"
        assert response == "X is Y."
        assert cp_id == "cp-abc"
