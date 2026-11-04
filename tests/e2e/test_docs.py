"""E2E tests for GET /api/docs/{slug} and /api/docs/troubleshoot.

Verifies the allowlist enforcement, path-traversal / separator
rejection, and the success path for a known doc. Reads the real
``docs/`` tree so the assertion is an end-to-end round trip.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rlmstudio.server.routes.docs import _DOCS_ALLOWLIST, _DOCS_DIR, _TROUBLESHOOT_FILE

pytestmark = [pytest.mark.e2e]


class TestDocsEndpoint:
    """GET /api/docs/{slug}."""

    def test_returns_200_with_slug_and_content_for_allowlisted_doc(
        self, client: TestClient
    ) -> None:
        resp = client.get("/api/docs/rlm-concepts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["slug"] == "rlm-concepts"
        assert isinstance(data["content"], str)
        assert len(data["content"]) > 0

    def test_content_matches_file_on_disk(self, client: TestClient) -> None:
        expected = (_DOCS_DIR / "rlm-concepts.md").read_text(encoding="utf-8")
        data = client.get("/api/docs/rlm-concepts").json()
        assert data["content"] == expected

    def test_all_allowlisted_slugs_resolve(self, client: TestClient) -> None:
        # Every entry in the allowlist must point at a readable file.
        for slug in _DOCS_ALLOWLIST:
            resp = client.get(f"/api/docs/{slug}")
            assert resp.status_code == 200, f"slug={slug} failed: {resp.text}"
            assert resp.json()["slug"] == slug

    def test_unknown_slug_returns_404(self, client: TestClient) -> None:
        resp = client.get("/api/docs/does-not-exist")
        assert resp.status_code == 404

    def test_slug_with_dot_dot_returns_404(self, client: TestClient) -> None:
        # Traversal attempt: /api/docs/..%2F..%2Fetc%2Fpasswd decoded.
        # Starlette decodes the path param before our handler, so we
        # exercise the literal form that would slip past the router.
        resp = client.get("/api/docs/..")
        assert resp.status_code == 404

    def test_slug_with_slash_returns_404(self, client: TestClient) -> None:
        # URL-encoded slash should not match an allowlisted slug.
        resp = client.get("/api/docs/hosts%2Follama")
        assert resp.status_code == 404

    def test_slug_with_uppercase_returns_404(self, client: TestClient) -> None:
        # Slug regex is lowercase-only; protects against case-insensitive
        # filesystem collisions on macOS/Windows.
        resp = client.get("/api/docs/RLM-Concepts")
        assert resp.status_code == 404

    def test_slug_with_dot_returns_404(self, client: TestClient) -> None:
        resp = client.get("/api/docs/rlm-concepts.md")
        assert resp.status_code == 404

    def test_allowlisted_filenames_exist_on_disk(self) -> None:
        # Structural invariant — every allowlist entry must be a real file.
        for slug, filename in _DOCS_ALLOWLIST.items():
            path = _DOCS_DIR / filename
            assert path.is_file(), f"allowlist entry {slug!r} -> {filename!r} missing"

    def test_project_root_resolves_to_repo_with_docs_dir(self) -> None:
        # Guards against a regression in the parents[N] math.
        assert _DOCS_DIR.is_dir()
        assert (_DOCS_DIR / "rlm-concepts.md").is_file()

    def test_docs_root_is_not_above_repo(self) -> None:
        # Sanity: resolved docs dir does not escape to the filesystem root.
        assert str(_DOCS_DIR.resolve()) != "/"
        # The docs dir must live directly under the repo root — identified by
        # a marker file, not by the checkout directory's name — unless a
        # release build copied it into the package (smoke-install.sh does).
        resolved = _DOCS_DIR.resolve()
        assert (resolved.parent / "pyproject.toml").is_file() or resolved.name == "_docs"

    def test_an_installed_package_reads_its_bundled_docs(self, tmp_path: Path) -> None:
        """A wheel has no repository docs/: the release build copies it into
        rlmstudio/_docs/, and the route must read that copy (Learn was empty
        in the installed 1.0.0rc1)."""
        from rlmstudio.server.routes.docs import _resolve_docs_dir

        routes = tmp_path / "site-packages" / "rlmstudio" / "server" / "routes"
        routes.mkdir(parents=True)
        packaged = tmp_path / "site-packages" / "rlmstudio" / "_docs"
        packaged.mkdir()
        (packaged / "troubleshoot.yaml").write_text("[]\n", encoding="utf-8")

        assert _resolve_docs_dir(routes / "docs.py") == packaged

    def test_a_placeholder_only_package_falls_back_to_repo_docs(self, tmp_path: Path) -> None:
        from rlmstudio.server.routes.docs import _resolve_docs_dir

        routes = tmp_path / "repo" / "src" / "rlmstudio" / "server" / "routes"
        routes.mkdir(parents=True)
        (tmp_path / "repo" / "src" / "rlmstudio" / "_docs").mkdir()

        assert _resolve_docs_dir(routes / "docs.py") == tmp_path / "repo" / "docs"


class TestDocsPathTraversalDefense:
    """The endpoint's second line of defense: resolved-path containment."""

    def test_symlink_pointing_outside_docs_dir_is_rejected(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Simulate a poisoned allowlist entry whose filename resolves,
        # via symlink, outside docs/. The containment check should 404.
        outside = tmp_path / "outside.md"
        outside.write_text("secret content", encoding="utf-8")

        link = _DOCS_DIR / "__test_symlink.md"
        try:
            link.symlink_to(outside)
        except OSError:
            pytest.skip("symlink creation unsupported on this platform")
        try:
            # Poison the allowlist for the duration of this test.
            from rlmstudio.server.routes import docs as docs_mod

            monkeypatch.setitem(docs_mod._DOCS_ALLOWLIST, "test-symlink", "__test_symlink.md")
            resp = client.get("/api/docs/test-symlink")
            assert resp.status_code == 404
        finally:
            link.unlink(missing_ok=True)


class TestTroubleshootEndpoint:
    """GET /api/docs/troubleshoot — parsed YAML as JSON."""

    def test_returns_200_with_entries_list(self, client: TestClient) -> None:
        resp = client.get("/api/docs/troubleshoot")
        assert resp.status_code == 200
        data = resp.json()
        assert "entries" in data
        assert isinstance(data["entries"], list)
        assert len(data["entries"]) > 0

    def test_every_entry_has_the_expected_shape(self, client: TestClient) -> None:
        data = client.get("/api/docs/troubleshoot").json()
        for entry in data["entries"]:
            assert isinstance(entry["id"], str)
            assert entry["id"]
            assert isinstance(entry["title"], str)
            assert entry["title"]
            assert isinstance(entry["symptom"], str)
            assert isinstance(entry["cause"], str)
            assert entry["category"] in {
                "Setup",
                "Provider",
                "Compare",
                "Judge",
                "Budget",
                "Runtime",
            }
            assert isinstance(entry["fix"], list)
            assert isinstance(entry["seealso"], list)

    def test_ids_are_unique(self, client: TestClient) -> None:
        ids = [e["id"] for e in client.get("/api/docs/troubleshoot").json()["entries"]]
        assert len(ids) == len(set(ids))

    def test_seed_entries_cover_the_pitch_list(self, client: TestClient) -> None:
        # The pitch §Seed Entries enumerates five required rows; verify
        # they ship so the V1 acceptance criteria on Troubleshoot hold.
        ids = {e["id"] for e in client.get("/api/docs/troubleshoot").json()["entries"]}
        required = {
            "anthropic-empty-response",
            "judge-flat-at-5",
            "compare-stuck-at-zero",
            "ollama-model-not-found",
            "token-budget-exhausted",
        }
        assert required.issubset(ids), f"missing seed entries: {required - ids}"

    def test_troubleshoot_route_takes_precedence_over_slug_route(self, client: TestClient) -> None:
        # Regression guard: if the /{slug} route were registered first,
        # GET /api/docs/troubleshoot would 404 because "troubleshoot"
        # is not in _DOCS_ALLOWLIST. The 200 here proves the specific
        # route beats the catch-all in FastAPI's match order.
        resp = client.get("/api/docs/troubleshoot")
        assert resp.status_code == 200

    def test_rejects_invalid_yaml_with_500(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        bad = tmp_path / "troubleshoot.yaml"
        bad.write_text("this: is: not: valid: yaml: [\n", encoding="utf-8")
        from rlmstudio.server.routes import docs as docs_mod

        monkeypatch.setattr(docs_mod, "_TROUBLESHOOT_FILE", bad)
        resp = client.get("/api/docs/troubleshoot")
        assert resp.status_code == 500

    def test_rejects_non_list_yaml_with_500(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        bad = tmp_path / "troubleshoot.yaml"
        bad.write_text("top-level: not a list\n", encoding="utf-8")
        from rlmstudio.server.routes import docs as docs_mod

        monkeypatch.setattr(docs_mod, "_TROUBLESHOOT_FILE", bad)
        resp = client.get("/api/docs/troubleshoot")
        assert resp.status_code == 500

    def test_rejects_entry_with_unknown_category_via_pydantic(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        bad = tmp_path / "troubleshoot.yaml"
        bad.write_text(
            "- id: bad\n"
            "  title: Bad entry\n"
            "  symptom: x\n"
            "  cause: y\n"
            "  category: NotARealCategory\n"
            "  fix: []\n",
            encoding="utf-8",
        )
        from rlmstudio.server.routes import docs as docs_mod

        monkeypatch.setattr(docs_mod, "_TROUBLESHOOT_FILE", bad)
        resp = client.get("/api/docs/troubleshoot")
        assert resp.status_code == 500

    def test_rejects_entry_without_fix_field_via_pydantic(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # `fix` is load-bearing — without remediation steps the card
        # ships nothing actionable. The schema must treat it as required.
        bad = tmp_path / "troubleshoot.yaml"
        bad.write_text(
            "- id: no-fix\n  title: Missing fix\n  symptom: x\n  cause: y\n  category: Runtime\n",
            encoding="utf-8",
        )
        from rlmstudio.server.routes import docs as docs_mod

        monkeypatch.setattr(docs_mod, "_TROUBLESHOOT_FILE", bad)
        resp = client.get("/api/docs/troubleshoot")
        assert resp.status_code == 500

    def test_missing_source_file_returns_500(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        nowhere = tmp_path / "does-not-exist.yaml"
        from rlmstudio.server.routes import docs as docs_mod

        monkeypatch.setattr(docs_mod, "_TROUBLESHOOT_FILE", nowhere)
        resp = client.get("/api/docs/troubleshoot")
        assert resp.status_code == 500

    def test_source_file_exists_on_disk(self) -> None:
        # Structural invariant: repo ships the file the endpoint points at.
        assert _TROUBLESHOOT_FILE.is_file()
