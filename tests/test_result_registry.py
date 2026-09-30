"""Tests for the controller-side result registry (specs/rlm-working-state, AC-3/AC-3a)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlmstudio.application.services.execution_preview import format_preview, truncation_marker
from rlmstudio.application.services.result_registry import (
    CHECKPOINT_EVERY_CHARS,
    ResultRegistry,
    canonical_text,
)

CJK = "修复登录故障，然后再试一次。"
EMOJI = "🚀🧪✅"


def _mixed_text(n_chars: int) -> str:
    """Deterministic text mixing ASCII, CJK and 4-byte emoji."""
    unit = "abc " + CJK + " " + EMOJI + " "
    reps = n_chars // len(unit) + 1
    return (unit * reps)[:n_chars]


class TestIdsAndLast:
    def test_ids_are_contiguous_and_last_tracks_newest(self) -> None:
        with ResultRegistry() as reg:
            assert reg.last is None
            a = reg.register("first", kind="inspect", step=3)
            b = reg.register(["x", "y"], kind="subcall", step=9)
            assert (a.id, b.id) == ("r1", "r2")
            assert reg.last is not None
            assert reg.last.id == "r2"
            assert reg.ids() == ["r1", "r2"]

    def test_ids_do_not_depend_on_step_numbers(self) -> None:
        with ResultRegistry() as reg:
            reg.register("a", kind="inspect", step=7)
            ref = reg.register("b", kind="inspect", step=42)
            assert ref.id == "r2"
            assert ref.step == 42

    def test_reads_never_change_last(self) -> None:
        with ResultRegistry() as reg:
            reg.register("0123456789" * 5, kind="inspect", step=1)
            first = reg.read("last", start=0, end=10)
            second = reg.read("last", start=10, end=20)
            assert first.startswith("0123456789")
            assert second.startswith("0123456789")
            assert reg.last is not None
            assert reg.last.id == "r1"
            assert reg.ids() == ["r1"]


class TestCanonicalText:
    def test_str_unchanged_list_as_stable_json(self) -> None:
        assert canonical_text("héllo") == "héllo"
        assert canonical_text(["a", "é"]) == json.dumps(["a", "é"], ensure_ascii=False, indent=2)
        assert canonical_text({"k": 1}) == '{\n  "k": 1\n}'

    def test_list_result_sliced_by_character(self) -> None:
        with ResultRegistry() as reg:
            ref = reg.register(["chunk one", "chunk two"], kind="inspect", step=1)
            text = reg.get(ref.id)
            assert text == '[\n  "chunk one",\n  "chunk two"\n]'
            body = reg.read(ref.id, start=4, end=15).split("\n... (showing chars")[0]
            assert body == text[4:15]


class TestReadSemantics:
    def test_slice_clamps_and_supports_negative_offsets(self) -> None:
        with ResultRegistry() as reg:
            ref = reg.register("abcdef", kind="inspect", step=1)
            assert reg.read(ref.id, start=-2) == "ef"
            assert reg.read(ref.id, start=4, end=100) == "ef"
            assert reg.read(ref.id, start=9, end=12) == ""

    def test_footer_names_id_and_next_offset(self) -> None:
        with ResultRegistry() as reg:
            ref = reg.register("x" * 100, kind="inspect", step=1)
            out = reg.read(ref.id, start=0, end=None, max_chars=40)
            assert out.startswith("x" * 40)
            assert f"read_result('{ref.id}', start=40)" in out
            assert "of 100" in out
            tail = reg.read(ref.id, start=60)
            assert tail == "x" * 40  # no footer when the read reaches the end

    def test_unknown_and_evicted_names_return_error_text(self) -> None:
        with ResultRegistry() as reg:
            assert reg.read("r9").startswith("Error: unknown result 'r9'")
            reg.register("a", kind="inspect", step=1)
            assert "r1..r1" in reg.read("nope")


class TestMemoryBound:
    def test_large_result_is_spilled_not_dropped(self, tmp_path: Path) -> None:
        text = _mixed_text(5000)
        with ResultRegistry(spill_result_above_bytes=1024, scratch_dir=tmp_path) as reg:
            ref = reg.register(text, kind="inspect", step=1)
            assert ref.spilled is True
            assert reg.in_memory_bytes == 0
            assert reg.spilled_bytes == len(text.encode("utf-8"))
            assert reg.get(ref.id) == text
            assert list(tmp_path.iterdir())  # something on disk

    def test_in_memory_total_never_exceeds_cap(self, tmp_path: Path) -> None:
        peak = 0

        def account(in_mem: int, _spilled: int) -> None:
            nonlocal peak
            peak = max(peak, in_mem)

        cap = 3000
        with ResultRegistry(
            spill_result_above_bytes=2000,
            max_registry_bytes=cap,
            scratch_dir=tmp_path,
            on_account=account,
        ) as reg:
            texts = [_mixed_text(900) for _ in range(6)]
            refs = [reg.register(t, kind="inspect", step=i) for i, t in enumerate(texts, 1)]
            assert peak <= cap
            assert reg.in_memory_bytes <= cap
            # Oldest results were spilled, nothing was lost.
            assert refs[0].id == "r1"
            assert reg.describe("r1") is not None
            assert reg.describe("r1").spilled is True  # type: ignore[union-attr]
            for ref, text in zip(refs, texts, strict=True):
                assert reg.get(ref.id) == text

    def test_last_alone_over_cap_is_spilled(self, tmp_path: Path) -> None:
        with ResultRegistry(
            spill_result_above_bytes=10_000, max_registry_bytes=500, scratch_dir=tmp_path
        ) as reg:
            ref = reg.register("y" * 800, kind="inspect", step=1)
            assert reg.in_memory_bytes == 0
            assert reg.describe(ref.id).spilled is True  # type: ignore[union-attr]
            assert reg.get(ref.id) == "y" * 800


class TestSpillBound:
    def test_oldest_spilled_evicted_never_last(self, tmp_path: Path) -> None:
        with ResultRegistry(
            spill_result_above_bytes=10, max_spill_bytes=250, scratch_dir=tmp_path
        ) as reg:
            r1 = reg.register("a" * 100, kind="inspect", step=1)
            r2 = reg.register("b" * 100, kind="inspect", step=2)
            r3 = reg.register("c" * 100, kind="inspect", step=3)
            assert reg.spilled_bytes <= 250
            assert reg.read(r1.id).startswith("Error: r1 was evicted")
            assert "step 3" in reg.read(r1.id)
            assert reg.get(r2.id) == "b" * 100
            assert reg.get(r3.id) == "c" * 100
            assert reg.last is not None
            assert reg.last.id == r3.id

    def test_single_result_over_spill_budget_is_truncated_and_flagged(self, tmp_path: Path) -> None:
        with ResultRegistry(
            spill_result_above_bytes=10, max_spill_bytes=64, scratch_dir=tmp_path
        ) as reg:
            ref = reg.register(_mixed_text(500), kind="inspect", step=1)
            assert ref.truncated is True
            assert ref.byte_length <= 64
            text = reg.get(ref.id)
            assert text is not None
            assert len(text.encode("utf-8")) <= 64
            # A cut never leaves a broken character behind.
            text.encode("utf-8").decode("utf-8")


class TestSpilledCharacterOffsets:
    """AC-3a: character-offset reads on spilled non-ASCII content are exact."""

    @pytest.mark.parametrize(
        "length", [1_000, CHECKPOINT_EVERY_CHARS + 777, 3 * CHECKPOINT_EVERY_CHARS]
    )
    def test_spilled_reads_match_in_memory_slices(self, tmp_path: Path, length: int) -> None:
        text = _mixed_text(length)
        with ResultRegistry(spill_result_above_bytes=64, scratch_dir=tmp_path) as reg:
            ref = reg.register(text, kind="inspect", step=1)
            assert ref.spilled
            probes = [
                (0, 50),
                (17, 18),
                (length // 2, length // 2 + 333),
                (CHECKPOINT_EVERY_CHARS - 5, CHECKPOINT_EVERY_CHARS + 5),
                (length - 40, None),
                (-25, None),
            ]
            for start, end in probes:
                expected = text[start:end] if end is not None else text[start:]
                got = reg.read(ref.id, start=start, end=end, max_chars=10**9)
                got_body = got.split("\n... (showing chars")[0]
                assert got_body == expected, (start, end)

    def test_head_of_spilled_result(self, tmp_path: Path) -> None:
        text = _mixed_text(2000)
        with ResultRegistry(spill_result_above_bytes=64, scratch_dir=tmp_path) as reg:
            ref = reg.register(text, kind="inspect", step=1)
            assert reg.head(ref.id, 123) == text[:123]


class TestLifecycle:
    def test_close_removes_scratch_dir_and_is_idempotent(self, tmp_path: Path) -> None:
        reg = ResultRegistry(spill_result_above_bytes=1, scratch_dir=tmp_path)
        reg.register("spill me", kind="inspect", step=1)
        assert list(tmp_path.iterdir())
        reg.close()
        assert not list(tmp_path.iterdir())
        reg.close()
        with pytest.raises(RuntimeError):
            reg.register("late", kind="inspect", step=2)

    def test_context_manager_closes_on_exception(self, tmp_path: Path) -> None:
        def _use_and_fail() -> None:
            with ResultRegistry(spill_result_above_bytes=1, scratch_dir=tmp_path) as reg:
                reg.register("spill me", kind="inspect", step=1)
                raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            _use_and_fail()
        assert not list(tmp_path.iterdir())

    def test_invalid_limits_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            ResultRegistry(max_registry_bytes=0)


class TestPreview:
    def test_short_result_has_no_marker(self) -> None:
        with ResultRegistry() as reg:
            ref = reg.register("short", kind="inspect", step=1)
            assert format_preview(reg, ref, cap=100) == "short"

    def test_long_result_marker_names_id_and_offset(self) -> None:
        with ResultRegistry() as reg:
            ref = reg.register("z" * 300, kind="inspect", step=1)
            out = format_preview(reg, ref, cap=100)
            assert out.startswith("z" * 100)
            assert out.endswith(truncation_marker(ref, 100))
            assert "300 chars total" in out
            assert "read_result('r1', start=100)" in out

    def test_marker_mentions_spill_truncation(self, tmp_path: Path) -> None:
        with ResultRegistry(
            spill_result_above_bytes=10, max_spill_bytes=64, scratch_dir=tmp_path
        ) as reg:
            ref = reg.register("w" * 500, kind="inspect", step=1)
            assert "cut at the spill budget" in format_preview(reg, ref, cap=10)
