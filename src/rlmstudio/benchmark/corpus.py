# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Materialise benchmark case content and programmatic ground truths.

A benchmark YAML describes *how to obtain* a case's document instead of
embedding it, so the tracked dataset stays a few kilobytes while the runs use
5K–150K-token documents.  Three kinds of ``content_spec`` are supported:

``generator``
    A deterministic synthetic technical document (seeded, ``chars/4`` token
    estimate so the output is identical on every machine) with facts planted
    at known positions.  Contamination-free: a model that has not read the
    document cannot know a planted fact.
``content_file``
    A file in the repository, relative to the YAML — the project's own docs.
``source_url`` + ``sha256``
    A public text (Project Gutenberg, RFC Editor) fetched into the corpus
    directory on demand and verified against its pinned digest.

``expected_spec`` derives the expected answer from the materialised text —
``count_word`` / ``count_sentence`` — so even a public novel yields an exact,
contamination-free aggregation target.
"""

from __future__ import annotations

import hashlib
import random
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .dataset import BenchmarkCase, BenchmarkDataset

CONTENT_SPEC_KEYS: frozenset[str] = frozenset(
    {"content_file", "generator", "source_url", "sha256", "filename"}
)
GENERATOR_SYNTHETIC = "synthetic"

SPEC_KEY_TOKENS = "tokens"
SPEC_KEY_SEED = "seed"
SPEC_KEY_TITLE = "title"
SPEC_KEY_FACTS = "facts"
SPEC_KEY_REPEAT = "repeat"
FACT_POSITION_MIDDLE = "middle"
FACT_POSITION_START = "start"
FACT_POSITION_END = "end"

EXPECTED_COUNT_WORD = "count_word"
EXPECTED_COUNT_SENTENCE = "count_sentence"

_CHARS_PER_TOKEN = 4

# The same vocabulary as scripts/testdata/make_test_corpus.py, so benchmark
# documents read like the release test corpus.  (That script is a release tool
# outside this package's allowlist, hence the copy rather than an import.)
_TOPICS = [
    "ingest pipeline",
    "retention policy",
    "on-call rotation",
    "schema migration",
    "cache invalidation",
    "rate limiting",
    "backfill job",
    "index compaction",
    "quota accounting",
    "shadow traffic",
]
_VERBS = [
    "records",
    "reconciles",
    "drains",
    "replays",
    "throttles",
    "partitions",
    "checkpoints",
    "rebalances",
]
_NOUNS = [
    "the write-ahead log",
    "the staging bucket",
    "the consumer group",
    "the replica set",
    "the audit trail",
    "the dead-letter queue",
    "the shard map",
    "the retry budget",
]


class CorpusError(ValueError):
    """A case's content could not be materialised as specified."""


@dataclass
class MaterialiseReport:
    """What :func:`materialise` did: which cases got content, which were skipped and why."""

    materialised: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Synthetic documents
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Deterministic ``chars/4`` estimate (no tokenizer dependency)."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _sentence(rng: random.Random) -> str:
    return (
        f"The {rng.choice(_TOPICS)} {rng.choice(_VERBS)} {rng.choice(_NOUNS)} "
        f"every {rng.randint(2, 90)} minutes."
    )


def _section(rng: random.Random, index: int) -> list[str]:
    blocks = [f"## Section {index}: {_TOPICS[index % len(_TOPICS)].title()}\n"]
    for _ in range(rng.randint(3, 6)):
        blocks.append(" ".join(_sentence(rng) for _ in range(rng.randint(4, 9))) + "\n")
    return blocks


def _insert_at(text: str, sentence: str, position: str) -> str:
    """Insert *sentence* as its own paragraph at the start, middle or end."""
    if position == FACT_POSITION_START:
        head, sep, tail = text.partition("\n\n")
        return f"{head}{sep}{sentence}\n\n{tail}" if sep else f"{sentence}\n\n{text}"
    if position == FACT_POSITION_END:
        return f"{text.rstrip()}\n\n{sentence}\n"
    if position == FACT_POSITION_MIDDLE:
        # Land between paragraphs near the midpoint so neither a head-read
        # nor a tail-read sees it.
        midpoint = len(text) // 2
        cut = text.find("\n\n", midpoint)
        if cut == -1:
            cut = text.rfind("\n\n", 0, midpoint)
        if cut == -1:
            return f"{text}\n\n{sentence}\n"
        return f"{text[:cut]}\n\n{sentence}{text[cut:]}"
    raise CorpusError(f"Unknown fact position {position!r}")


def _spread(text: str, sentence: str, times: int, rng: random.Random) -> str:
    """Insert *sentence* ``times`` times at distinct paragraph boundaries."""
    boundaries = [m.start() for m in re.finditer(r"\n\n", text)]
    if times > len(boundaries):
        raise CorpusError(f"Cannot spread {times} sentences over {len(boundaries)} paragraphs")
    chosen = sorted(rng.sample(boundaries, times), reverse=True)
    for cut in chosen:
        text = f"{text[:cut]}\n\n{sentence}{text[cut:]}"
    return text


def generate_synthetic(spec: dict[str, Any]) -> str:
    """Build the document described by a ``generator: synthetic`` spec.

    Keys: ``tokens`` (target size), ``seed``, ``title``, ``facts`` (list of
    ``{sentence, position}``), ``repeat`` (``{sentence, times}``).  Facts and
    repeats are inserted *after* the body is grown, so the body itself is
    identical for a given ``(tokens, seed, title)``.
    """
    target = int(spec.get(SPEC_KEY_TOKENS, 5_000))
    seed = int(spec.get(SPEC_KEY_SEED, 0))
    title = str(spec.get(SPEC_KEY_TITLE, "Platform Operations Handbook"))
    rng = random.Random(seed)

    blocks = [f"# {title}\n"]
    text = "\n".join(blocks)
    index = 0
    while estimate_tokens(text) < target:
        blocks.extend(_section(rng, index))
        index += 1
        text = "\n".join(blocks)

    for fact in spec.get(SPEC_KEY_FACTS) or []:
        text = _insert_at(
            text, str(fact["sentence"]), str(fact.get("position", FACT_POSITION_MIDDLE))
        )
    repeat = spec.get(SPEC_KEY_REPEAT)
    if repeat:
        text = _spread(text, str(repeat["sentence"]), int(repeat["times"]), rng)
    return text


# ---------------------------------------------------------------------------
# Public texts
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 — pinned by sha256
        dest.write_bytes(response.read())


def _public_text(spec: dict[str, Any], corpus_dir: Path, *, fetch: bool) -> str | None:
    """Return the text for a ``source_url`` spec, or ``None`` when absent and not fetching."""
    url = str(spec["source_url"])
    filename = str(spec.get("filename") or url.rsplit("/", 1)[-1])
    expected_sha = spec.get("sha256")
    path = corpus_dir / filename
    if not path.exists():
        if not fetch:
            return None
        _fetch(url, path)
    if expected_sha:
        actual = _sha256(path)
        if actual != expected_sha:
            raise CorpusError(
                f"{filename}: sha256 mismatch (expected {expected_sha[:12]}…, got {actual[:12]}…)"
            )
    return path.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Expected answers derived from the text
# ---------------------------------------------------------------------------


def count_word(text: str, word: str) -> int:
    """Whole-word, case-insensitive occurrences."""
    return len(re.findall(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE))


def derive_expected(text: str, spec: dict[str, Any]) -> str:
    if EXPECTED_COUNT_WORD in spec:
        return str(count_word(text, str(spec[EXPECTED_COUNT_WORD])))
    if EXPECTED_COUNT_SENTENCE in spec:
        return str(text.count(str(spec[EXPECTED_COUNT_SENTENCE])))
    raise CorpusError(f"Unknown expected_spec {sorted(spec)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def materialise_case(
    case: BenchmarkCase,
    *,
    base_dir: Path,
    corpus_dir: Path,
    fetch: bool = False,
) -> str | None:
    """Fill ``case.content`` (and a derived ``expected_answer``) in place.

    Returns a skip reason instead of raising when a public text is absent and
    ``fetch`` is False, so a run can proceed on the cases it has.
    """
    spec = case.content_spec or {}
    if not case.content:
        if "generator" in spec:
            if spec["generator"] != GENERATOR_SYNTHETIC:
                raise CorpusError(f"{case.id}: unknown generator {spec['generator']!r}")
            case.content = generate_synthetic(spec)
        elif "content_file" in spec:
            case.content = (base_dir / str(spec["content_file"])).read_text(encoding="utf-8")
        elif "source_url" in spec:
            text = _public_text(spec, corpus_dir, fetch=fetch)
            if text is None:
                return f"public text not fetched (run with --fetch): {spec['source_url']}"
            case.content = text
        else:
            raise CorpusError(f"{case.id}: no content and no content_spec")
    if case.expected_spec:
        case.expected_answer = derive_expected(case.content, case.expected_spec)
    return None


def materialise(
    dataset: BenchmarkDataset,
    *,
    base_dir: Path,
    corpus_dir: Path,
    fetch: bool = False,
) -> MaterialiseReport:
    """Materialise every case; see :func:`materialise_case`."""
    report = MaterialiseReport()
    for case in dataset:
        reason = materialise_case(case, base_dir=base_dir, corpus_dir=corpus_dir, fetch=fetch)
        if reason is None:
            report.materialised.append(case.id)
        else:
            report.skipped[case.id] = reason
    return report
