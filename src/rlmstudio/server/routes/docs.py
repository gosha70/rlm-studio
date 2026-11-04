"""Docs endpoint — allowlisted markdown loader for the Learn tab.

``docs/`` is authoritative. Learn does not fork content; it reads
repo markdown at runtime via this allowlisted endpoint. An explicit
slug → filename mapping (``_DOCS_ALLOWLIST``) prevents arbitrary
file reads, and a resolved-path containment check provides
defense-in-depth against path traversal via symlinks.

See ``doc_internal/specs/learn-tab/pitch-learn-tab.md`` §Backend.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from rlmstudio.server.models import (
    DocResponse,
    TroubleshootEntry,
    TroubleshootResponse,
)

router = APIRouter()


def _resolve_docs_dir(module_file: Path | None = None) -> Path:
    """The packaged ``rlmstudio/_docs/`` copy when present, else the repo's ``docs/``.

    An installed wheel has no repository ``docs/``: the release build copies it
    into ``rlmstudio/_docs/`` (see ``RELEASING.md``). A source checkout keeps
    only the placeholder there and reads ``docs/`` four parents above this file
    (parents[0]=routes, [1]=server, [2]=rlmstudio, [3]=src, [4]=<repo root>).
    """
    here = (module_file or Path(__file__)).resolve()
    packaged = here.parents[2] / "_docs"
    if (packaged / "troubleshoot.yaml").is_file():
        return packaged
    return here.parents[4] / "docs"


_DOCS_DIR: Path = _resolve_docs_dir()

# Slug must match this regex before any lookup. Defense in depth against
# an accidentally loose allowlist; keeps slugs URL-clean.
_SLUG_PATTERN = re.compile(r"^[a-z0-9-]+$")

# Explicit slug → filename allowlist. Files live under _DOCS_DIR.
# Later steps (Cookbook) may extend this with hosts/<provider>.md
# entries; when they do, _SLUG_PATTERN should be updated alongside.
_DOCS_ALLOWLIST: dict[str, str] = {
    "rlm-concepts": "rlm-concepts.md",
    "rlm-prompt-tuning": "rlm-prompt-tuning.md",
    "rlm-studio-guide": "rlm-studio-guide.md",
    "rlm-studio-design-document": "RLM_Studio_Design_Document.md",
    "lessons-from-ai-copilots": "lessons-from-ai-copilots.md",
    # Cookbook provider guides. Slug uses flat "hosts-<name>" form so
    # the slug regex ^[a-z0-9-]+$ stays simple; the filename resolves
    # into the nested docs/hosts/ directory.
    "hosts-ollama": "hosts/ollama.md",
    "hosts-lmstudio": "hosts/lmstudio.md",
    "hosts-vllm": "hosts/vllm.md",
    "hosts-dgx-spark": "hosts/dgx-spark.md",
    "hosts-openai": "hosts/openai.md",
    "hosts-anthropic": "hosts/anthropic.md",
    # hosts-groq intentionally omitted: the app's provider catalog does
    # not currently support a `groq` backend. When Groq support lands,
    # re-add this entry and the matching frontend catalog entry.
}


_TROUBLESHOOT_FILE: Path = _DOCS_DIR / "troubleshoot.yaml"


@router.get("/api/docs/troubleshoot")
async def get_troubleshoot() -> TroubleshootResponse:
    """Return parsed Troubleshoot entries as JSON.

    Registered BEFORE the ``/api/docs/{slug}`` catch-all so FastAPI's
    top-down route matcher lands here for the literal "troubleshoot"
    path instead of falling through to the slug handler.

    Pydantic validation on :class:`TroubleshootEntry` rejects any
    entry missing a required field or using an unsupported category
    — a stray edit in docs/troubleshoot.yaml will surface as a 500
    here rather than shipping broken rows to the client.
    """
    if not _TROUBLESHOOT_FILE.is_file():
        raise HTTPException(status_code=500, detail="Troubleshoot source missing")

    try:
        raw = _TROUBLESHOOT_FILE.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Failed to read troubleshoot") from exc

    try:
        parsed: Any = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=500, detail="Troubleshoot YAML parse error") from exc

    if not isinstance(parsed, list):
        raise HTTPException(status_code=500, detail="Troubleshoot YAML must be a list")

    entries: list[TroubleshootEntry] = []
    for row in parsed:
        if not isinstance(row, dict):
            raise HTTPException(status_code=500, detail="Troubleshoot entry must be a mapping")
        try:
            entries.append(TroubleshootEntry.model_validate(row))
        except ValidationError as exc:
            raise HTTPException(
                status_code=500, detail="Troubleshoot entry failed validation"
            ) from exc

    return TroubleshootResponse(entries=entries)


@router.get("/api/docs/{slug}")
async def get_doc(slug: str) -> DocResponse:
    """Return the markdown body of an allowlisted doc.

    Rejects anything not in the allowlist with 404 so the endpoint
    never leaks whether an unlisted file exists on disk. Also rejects
    resolved paths that escape ``docs/`` (symlink / traversal guard).
    """
    if not _SLUG_PATTERN.fullmatch(slug):
        raise HTTPException(status_code=404, detail="Doc not found")

    filename = _DOCS_ALLOWLIST.get(slug)
    if filename is None:
        raise HTTPException(status_code=404, detail="Doc not found")

    target = (_DOCS_DIR / filename).resolve()
    docs_root = _DOCS_DIR.resolve()
    # Containment check: target must be a descendant of docs_root. Blocks
    # resolution via symlinks that point outside the docs/ tree.
    try:
        target.relative_to(docs_root)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Doc not found") from exc

    if not target.is_file():
        raise HTTPException(status_code=404, detail="Doc not found")

    try:
        content = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Failed to read doc") from exc

    return DocResponse(slug=slug, content=content)
