"""`rlm-studio bench` end to end on the offline fakes (specs/benchmarks-v1 FR-1, FR-7)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlmstudio.application.sandbox_vars import MODE_DIRECT, MODE_RAG, MODE_RLM, MODE_RLM_OFFICIAL
from rlmstudio.benchmark.report import BENCH_END_MARKER, BENCH_START_MARKER
from rlmstudio.cli.main import main

REPO_ROOT = Path(__file__).resolve().parents[1]
LONGDOC = REPO_ROOT / "benchmarks" / "longdoc-v1.yaml"
ALL_ENGINES = ",".join([MODE_DIRECT, MODE_RAG, MODE_RLM, MODE_RLM_OFFICIAL])


def _run(argv: list[str]) -> int:
    code: int = main(["bench", *argv])
    return code


class TestDryRun:
    def test_end_to_end_writes_results_and_regenerates_the_page(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        page = tmp_path / "page.md"
        page.write_text(
            f"# Benchmarks\n\n{BENCH_START_MARKER}\nplaceholder\n{BENCH_END_MARKER}\n",
            encoding="utf-8",
        )

        code = _run(
            [
                "--config",
                str(LONGDOC),
                "--dry-run",
                "--limit",
                "2",
                "--providers",
                "fake/ok,fake/fail",
                "--engines",
                ALL_ENGINES,
                "--judge",
                "fake/judge",
                "--out",
                str(tmp_path / "out"),
                "--page",
                str(page),
                "--corpus-dir",
                str(tmp_path / "corpus"),
            ]
        )

        assert code == 0
        data = json.loads((tmp_path / "out" / "results.json").read_text())
        results = data["results"]
        assert results["metadata"]["dry_run"] is True
        assert results["judge_model"] == "fake/judge"
        assert len(results["case_ids"]) == 2
        # 2 cases × 2 providers × 4 engines, every engine through the real use cases
        assert len(results["outcomes"]) == 16
        engines_seen = {o["engine"] for o in results["outcomes"]}
        assert engines_seen == {MODE_DIRECT, MODE_RAG, MODE_RLM, MODE_RLM_OFFICIAL}
        ok = [o for o in results["outcomes"] if o["provider"] == "fake" and o["model"] == "ok"]
        failed = [o for o in results["outcomes"] if o["model"] == "fail"]
        assert all(o["success"] for o in ok)
        assert all(not o["success"] for o in failed)
        assert all(o["judge"] is not None for o in ok)
        assert all(o["judge"] is None for o in failed)
        # The public texts are skipped, not failed, without --fetch.
        assert results["skipped"]
        # Cells: one per (provider, engine).
        assert len(data["cells"]) == 8

        page_text = page.read_text(encoding="utf-8")
        assert "placeholder" not in page_text
        assert "`rlm_official`" in page_text
        assert (tmp_path / "out" / "results.md").read_text().count("### Per case") == 1

        out = capsys.readouterr().out
        assert "✓" in out
        assert "✗" in out
        assert "Regenerated" in out

    def test_accuracy_pass_path_is_exercised(self, tmp_path: Path) -> None:
        _run(
            [
                "--config",
                str(LONGDOC),
                "--dry-run",
                "--cases",
                "synthetic-5k-needle",
                "--providers",
                "fake/ok",
                "--engines",
                ALL_ENGINES,
                "--out",
                str(tmp_path / "out"),
                "--corpus-dir",
                str(tmp_path / "corpus"),
            ]
        )

        data = json.loads((tmp_path / "out" / "results.json").read_text())
        assert all(o["match_pass"] is True for o in data["results"]["outcomes"])
        assert all(c["accuracy"] == 1.0 for c in data["cells"])


class TestErrors:
    def test_unknown_engine_is_a_usage_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = _run(
            [
                "--config",
                str(LONGDOC),
                "--dry-run",
                "--providers",
                "fake/ok",
                "--engines",
                "magic",
                "--out",
                str(tmp_path),
            ]
        )

        assert code == 2
        assert "unknown engine(s) magic" in capsys.readouterr().err

    def test_page_without_markers_fails_cleanly(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        page = tmp_path / "page.md"
        page.write_text("no markers", encoding="utf-8")

        code = _run(
            [
                "--config",
                str(LONGDOC),
                "--dry-run",
                "--limit",
                "1",
                "--providers",
                "fake/ok",
                "--engines",
                MODE_DIRECT,
                "--out",
                str(tmp_path / "out"),
                "--page",
                str(page),
                "--corpus-dir",
                str(tmp_path / "corpus"),
            ]
        )

        assert code == 1
        assert "missing" in capsys.readouterr().err

    def test_missing_config_fails_cleanly(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = _run(
            [
                "--config",
                str(tmp_path / "nope.yaml"),
                "--providers",
                "fake/ok",
                "--out",
                str(tmp_path),
            ]
        )

        assert code == 1
        assert "not found" in capsys.readouterr().err


class TestAStoppedRunKeepsWhatItPaidFor:
    """A real run is hours of provider calls; an error at the end must not void it."""

    def test_completed_cells_are_written_before_the_error_propagates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rlmstudio.benchmark import matrix_runner

        real_run_case = matrix_runner.MatrixBenchmarkRunner._run_case
        calls = {"n": 0}

        def _explode_on_the_second_case(self: object, case: object, rep: int) -> object:
            calls["n"] += 1
            if calls["n"] > 1:
                raise RuntimeError("provider melted down")
            return real_run_case(self, case, rep)  # type: ignore[arg-type]

        monkeypatch.setattr(
            matrix_runner.MatrixBenchmarkRunner, "_run_case", _explode_on_the_second_case
        )

        code = _run(
            [
                "--config",
                str(LONGDOC),
                "--dry-run",
                "--limit",
                "2",
                "--providers",
                "fake/ok",
                "--engines",
                MODE_DIRECT,
                "--out",
                str(tmp_path),
            ]
        )

        assert code == 1  # a clean failure, not a traceback

        partial = json.loads((tmp_path / "results.partial.json").read_text(encoding="utf-8"))
        results = partial["results"]
        assert results["outcomes"], "the first case's cells were paid for and must be kept"
        assert "melted down" in results["metadata"]["incomplete"]
        assert not (tmp_path / "results.json").exists()  # not a complete run
