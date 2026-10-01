# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Content navigation tools for exploring large text."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

import regex as regex_module


class PatternTimeoutError(Exception):
    """A model-written pattern used up the time allowed for one search.

    The pattern, not the document, is what makes this possible: a nested
    quantifier such as ``(a+)+$`` backtracks exponentially in the length of a
    single line, so 40 characters is enough to run for hours.  The stdlib
    engine holds the GIL throughout and cannot be interrupted by a signal or
    another thread, which is why the bound has to come from the engine itself.
    """

    def __init__(self, budget: float) -> None:
        self.budget = budget
        super().__init__(f"Pattern search timed out after {budget:.1f}s")


class _Matcher(Protocol):
    """The one method :func:`grep` needs from a compiled pattern."""

    def search(self, line: str) -> Any: ...


class _UnboundedMatcher:
    """The stdlib engine, for callers that are bounded by something else."""

    def __init__(self, compiled: re.Pattern[str]) -> None:
        self._compiled = compiled

    def search(self, line: str) -> Any:
        return self._compiled.search(line)


class _BoundedMatcher:
    """Spends at most ``budget`` seconds in the engine across every line.

    The budget covers the whole ``grep`` call rather than each line, which is
    what the sandbox's own timeout used to mean; a per-line limit would
    multiply by the line count.  ``regex`` is used instead of ``re`` because
    it checks a deadline inside its matching loop — and, as a bonus, it
    optimises away most classic blow-up patterns outright.
    """

    def __init__(self, compiled: Any, budget: float) -> None:
        self._compiled = compiled
        self._budget = budget
        self._spent = 0.0

    def search(self, line: str) -> Any:
        remaining = self._budget - self._spent
        if remaining <= 0:
            raise PatternTimeoutError(self._budget)
        started = time.perf_counter()
        try:
            return self._compiled.search(line, timeout=remaining)
        except TimeoutError as exc:
            raise PatternTimeoutError(self._budget) from exc
        finally:
            self._spent += time.perf_counter() - started


def _compile_matcher(pattern: str, *, ignore_case: bool, timeout: float | None) -> _Matcher:
    """Compile *pattern*, bounding its total match time when asked.

    Raises ``re.error`` for an invalid pattern either way, so the caller's
    existing "Invalid regex pattern" path is unchanged.
    """
    if timeout is None:
        flags = re.IGNORECASE if ignore_case else 0
        return _UnboundedMatcher(re.compile(pattern, flags))
    rx_flags = regex_module.IGNORECASE if ignore_case else 0
    try:
        compiled = regex_module.compile(pattern, rx_flags)
    except regex_module.error as exc:  # same shape as the stdlib error
        raise re.error(str(exc)) from exc
    return _BoundedMatcher(compiled, timeout)


def peek(content: str, start: int = 0, end: int | None = None, max_chars: int = 10000) -> str:
    """
    View a section of content by character positions.

    Args:
        content: The content to peek into
        start: Starting character position (0-indexed)
        end: Ending character position (exclusive). If None, goes to content end
        max_chars: Maximum characters to return (safety limit)

    Returns:
        Substring from start to end (limited by max_chars)

    Examples:
        >>> peek(P, 0, 100)  # First 100 chars
        >>> peek(P, 1000, 2000)  # Chars 1000-2000
        >>> peek(P, -100)  # Last 100 chars (Python slice semantics)
    """
    # Handle None end
    if end is None:
        end = len(content)

    # Bounds checking
    content_len = len(content)
    if start < 0:
        start = max(0, content_len + start)  # Handle negative indexing
    if end < 0:
        end = max(0, content_len + end)

    # Clamp to content bounds
    start = max(0, min(start, content_len))
    end = max(start, min(end, content_len))

    # Extract substring
    result = content[start:end]

    # Apply max_chars limit
    if len(result) > max_chars:
        result = (
            result[:max_chars]
            + f"\n... (truncated, showing {max_chars}/{len(content[start:end])} chars)"
        )

    return result


@dataclass(frozen=True)
class _FileSection:
    """Resolved file boundaries inside a flattened content string."""

    file_no: int
    name: str
    file_start: int
    content_start: int
    file_end_exclusive: int


_MULTI_FILE_INDEX_ENTRY_RE = re.compile(
    r'^\s*(?P<file_no>\d+)\.\s+"(?P<name>.+?)"\s+\('
    r"file_start=(?P<file_start>\d+), "
    r"content_start=(?P<content_start>\d+), "
    r"file_end_exclusive=(?P<file_end>\d+)\)$",
    flags=re.MULTILINE,
)
_SINGLE_FILE_HEADER_RE = re.compile(r"^\[File:\s*(?P<name>.+?)\]\n\n", flags=re.DOTALL)


def _iter_file_sections(content: str) -> list[_FileSection]:
    """Return file sections embedded in *content*.

    Supports:
    - multi-file content produced by chat._resolve_file_content()
    - single-file content prefixed with ``[File: name]``
    - raw content without headers (treated as file 1)
    """
    if content.startswith("[DOCUMENT INDEX"):
        index_end = content.find("[END DOCUMENT INDEX]")
        if index_end != -1:
            index_block = content[: index_end + len("[END DOCUMENT INDEX]")]
            sections = [
                _FileSection(
                    file_no=int(match.group("file_no")),
                    name=match.group("name"),
                    file_start=int(match.group("file_start")),
                    content_start=int(match.group("content_start")),
                    file_end_exclusive=int(match.group("file_end")),
                )
                for match in _MULTI_FILE_INDEX_ENTRY_RE.finditer(index_block)
            ]
            if sections:
                return sections

    single_match = _SINGLE_FILE_HEADER_RE.match(content)
    if single_match:
        return [
            _FileSection(
                file_no=1,
                name=single_match.group("name"),
                file_start=0,
                content_start=single_match.end(),
                file_end_exclusive=len(content),
            )
        ]

    return [
        _FileSection(
            file_no=1,
            name="document",
            file_start=0,
            content_start=0,
            file_end_exclusive=len(content),
        )
    ]


def _get_file_section(content: str, file_no: int) -> _FileSection:
    """Return the requested file section or raise ValueError."""
    if file_no <= 0:
        raise ValueError(f"file_no must be >= 1, got: {file_no}")

    for section in _iter_file_sections(content):
        if section.file_no == file_no:
            return section

    available = ", ".join(str(section.file_no) for section in _iter_file_sections(content))
    raise ValueError(f"Unknown file_no: {file_no}. Available file numbers: {available}")


def _get_file_content(content: str, file_no: int) -> tuple[_FileSection, str]:
    """Return ``(section, file_content)`` for *file_no*."""
    section = _get_file_section(content, file_no)
    return section, content[section.content_start : section.file_end_exclusive]


def _normalize_file_offset(
    section: _FileSection,
    offset: int | None,
    *,
    anchored_to_global_span: bool = False,
) -> int | None:
    """Normalize mixed absolute/file-relative offsets for file-scoped tools.

    Small models sometimes copy ``content_start`` / ``file_start`` values from
    the document index and pass them directly into ``peek_file()`` even though
    the tool expects file-relative offsets. This helper tolerates that by
    converting obvious document-global offsets back to file-relative values.
    """
    if offset is None:
        return None

    if offset in (
        section.file_start,
        section.content_start,
        section.file_end_exclusive,
    ):
        return offset - section.content_start

    if anchored_to_global_span and section.file_start <= offset <= section.file_end_exclusive:
        return offset - section.content_start

    return offset


def _normalize_line(line: str) -> str:
    """Normalize whitespace for heading and boilerplate detection."""
    return " ".join(line.split())


def _looks_like_page_number(line: str) -> bool:
    """Return True for isolated page-number-like strings."""
    compact = line.strip().replace(" ", "")
    if not compact:
        return False
    return bool(re.fullmatch(r"(?i)(page)?\d{1,4}|[ivxlcdm]{1,8}", compact))


def _is_probable_boilerplate_line(line: str) -> bool:
    """Return True for common PDF furniture / legal boilerplate lines."""
    normalized = _normalize_line(line)
    if not normalized:
        return True

    lowered = normalized.lower()
    if _looks_like_page_number(normalized):
        return True

    boilerplate_prefixes = (
        "licensed to ",
        "copyright ",
        "©",
        "manning publications",
        "for more information on this and other manning books",
        "printed in the united states of america",
        "development editor:",
        "technical development editor:",
        "production editor:",
        "copy editor:",
        "proofreader:",
        "technical proofreaders:",
        "typesetter and cover designer:",
        "special sales department",
        "po box ",
        "shelter island",
    )
    if lowered.startswith(boilerplate_prefixes):
        return True

    return False


def _looks_like_heading(line: str) -> bool:
    """Heuristic heading detector used by ``outline_file()``."""
    normalized = _normalize_line(line)
    if not normalized or _is_probable_boilerplate_line(normalized):
        return False

    if len(normalized) > 120:
        return False

    lowered = normalized.lower()
    heading_keywords = (
        "contents",
        "brief contents",
        "table of contents",
        "preface",
        "foreword",
        "introduction",
        "summary",
        "conclusion",
        "appendix",
        "chapter",
        "part ",
    )
    if lowered.startswith(heading_keywords):
        return True

    if re.match(r"^(chapter|part)\b", lowered):
        return True

    if re.match(r"^\d+(\.\d+)*\s+\S", normalized):
        return True

    if re.match(r"^[A-Z][A-Za-z0-9&'/:+,\-\s]{3,80}\s+\d{1,4}$", normalized):
        return True

    letters = [ch for ch in normalized if ch.isalpha()]
    if letters:
        uppercase_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
        if uppercase_ratio >= 0.85 and len(normalized) <= 90:
            return True

    return False


def grep(
    content: str,
    pattern: str,
    context_lines: int = 2,
    max_matches: int = 100,
    ignore_case: bool = False,
    use_regex: bool = False,
    timeout: float | None = None,
) -> str:
    """
    Search content for pattern and return matches with context.

    Args:
        content: The content to search
        pattern: Search pattern (literal string or regex)
        context_lines: Number of lines to show before/after each match
        max_matches: Maximum number of matches to return
        ignore_case: Case-insensitive search
        use_regex: Treat pattern as regex (default: literal string)
        timeout: Total seconds the regex engine may spend across all lines.
            Required when the caller runs in the controller process, where
            nothing else bounds a model-written pattern; ``None`` keeps the
            stdlib engine and relies on the caller's own limit (the sandbox).

    Returns:
        Formatted string with matches and line numbers

    Raises:
        PatternTimeoutError: The pattern exhausted ``timeout``.

    Examples:
        >>> grep(P, "error")  # Find "error" with 2 lines context
        >>> grep(P, r"\\d{4}-\\d{2}-\\d{2}", use_regex=True)  # Find dates
        >>> grep(P, "warning", context_lines=5, ignore_case=True)
    """
    lines = content.splitlines(keepends=True)

    # Prepare pattern
    if not use_regex:
        pattern = re.escape(pattern)  # Escape special chars for literal match

    try:
        matcher = _compile_matcher(pattern, ignore_case=ignore_case, timeout=timeout)
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"

    # Find matching lines
    matches: list[tuple[int, str, int]] = []
    char_offset = 0
    for line_no, line in enumerate(lines, start=1):
        if matcher.search(line):
            matches.append((line_no, line, char_offset))
            if len(matches) >= max_matches:
                break
        char_offset += len(line)

    if not matches:
        return f"No matches found for pattern: {pattern}"

    # Build output with context
    result_lines: list[str] = []
    result_lines.append(f"Found {len(matches)} match(es):\n")

    for match_line_no, _match_line, match_offset in matches:
        # Calculate context range
        start_line = max(1, match_line_no - context_lines)
        end_line = min(len(lines), match_line_no + context_lines)

        # Add separator
        result_lines.append(f"\n{'=' * 60}")
        result_lines.append(f"Match at line {match_line_no} (char_offset {match_offset}):")
        result_lines.append(f"{'=' * 60}")

        # Add context and match
        for line_no in range(start_line, end_line + 1):
            line = lines[line_no - 1].rstrip("\r\n")  # Convert to 0-indexed
            prefix = ">>> " if line_no == match_line_no else "    "
            result_lines.append(f"{prefix}{line_no:4d}: {line}")

    return "\n".join(result_lines)


def peek_file(
    content: str,
    file_no: int,
    start: int = 0,
    end: int | None = None,
    max_chars: int = 10000,
) -> str:
    """
    View a section of a specific attached file using file-relative offsets.

    Args:
        content: The flattened content string ``P``
        file_no: 1-based file number from the document index
        start: Starting character position inside the file content
        end: Ending character position inside the file content
        max_chars: Maximum characters to return

    Returns:
        Substring from the selected file
    """
    section, file_content = _get_file_content(content, file_no)
    normalized_start = _normalize_file_offset(section, start)
    anchored = normalized_start != start
    normalized_end = _normalize_file_offset(
        section,
        end,
        anchored_to_global_span=anchored,
    )
    return peek(file_content, start=normalized_start or 0, end=normalized_end, max_chars=max_chars)


def grep_file(
    content: str,
    file_no: int,
    pattern: str,
    context_lines: int = 2,
    max_matches: int = 100,
    ignore_case: bool = False,
    use_regex: bool = False,
    timeout: float | None = None,
) -> str:
    """
    Search within a specific file.

    Returned line numbers and ``char_offset`` values are relative to that file's
    content, making them safe follow-up inputs for ``peek_file()``.

    ``timeout`` bounds the regex engine exactly as it does for :func:`grep`.
    """
    section, file_content = _get_file_content(content, file_no)
    result = grep(
        file_content,
        pattern=pattern,
        context_lines=context_lines,
        max_matches=max_matches,
        ignore_case=ignore_case,
        use_regex=use_regex,
        timeout=timeout,
    )
    header = (
        f'File {section.file_no}: "{section.name}" '
        "(line numbers and char_offset values are relative to this file)\n"
    )
    return header + result


def outline_file(
    content: str,
    file_no: int,
    max_lines: int = 40,
    max_chars: int = 8000,
) -> str:
    """
    Extract a compact structural outline for one file.

    The outline favors titles, table-of-contents entries, chapter headings, and
    section headers while skipping obvious PDF boilerplate.
    """
    if max_lines <= 0:
        raise ValueError(f"max_lines must be positive, got: {max_lines}")

    section, file_content = _get_file_content(content, file_no)
    sample = file_content[: min(len(file_content), 120_000)]
    raw_lines = sample.splitlines()

    title = next(
        (_normalize_line(line) for line in raw_lines if not _is_probable_boilerplate_line(line)),
        section.name,
    )

    outline_lines: list[str] = []
    seen: set[str] = set()
    for raw_line in raw_lines:
        line = _normalize_line(raw_line)
        if not _looks_like_heading(line):
            continue
        if line in seen:
            continue
        outline_lines.append(line)
        seen.add(line)
        if len(outline_lines) >= max_lines:
            break

    if not outline_lines:
        # Fall back to the first substantive lines when a clean outline is missing.
        for raw_line in raw_lines:
            line = _normalize_line(raw_line)
            if not line or _is_probable_boilerplate_line(line):
                continue
            outline_lines.append(line)
            if len(outline_lines) >= min(max_lines, 12):
                break

    result_lines = [
        f'File {section.file_no}: "{section.name}"',
        f"Title: {title}",
        f"Outline highlights (up to {max_lines} lines):",
    ]
    result_lines.extend(f"- {line}" for line in outline_lines)
    result = "\n".join(result_lines)
    if len(result) > max_chars:
        result = result[:max_chars] + f"\n... (truncated, showing {max_chars}/{len(result)} chars)"
    return result


def chunk(
    content: str, size: int = 1000, overlap: int = 0, by: str = "chars", max_chunks: int = 100
) -> list[str]:
    """
    Split content into chunks for systematic exploration.

    Args:
        content: The content to chunk
        size: Chunk size (chars or lines depending on 'by')
        overlap: Overlap between chunks (0 = no overlap)
        by: 'chars' or 'lines' - chunking strategy
        max_chunks: Maximum number of chunks to return

    Returns:
        List of content chunks

    Examples:
        >>> chunks = chunk(P, size=1000, by='chars')  # 1000-char chunks
        >>> chunks = chunk(P, size=50, by='lines')    # 50-line chunks
        >>> chunks = chunk(P, size=500, overlap=50)   # With 50-char overlap
    """
    if by not in ("chars", "lines"):
        raise ValueError(f"Invalid 'by' parameter: {by}. Must be 'chars' or 'lines'")

    if size <= 0:
        raise ValueError(f"Size must be positive, got: {size}")

    if overlap < 0 or overlap >= size:
        raise ValueError(f"Overlap must be 0 <= overlap < size, got: {overlap}")

    chunks: list[str] = []

    if by == "chars":
        # Character-based chunking
        step = size - overlap
        for i in range(0, len(content), step):
            chunk_text = content[i : i + size]
            chunks.append(chunk_text)

            if len(chunks) >= max_chunks:
                break

    else:  # by == 'lines'
        # Line-based chunking
        lines = content.splitlines(keepends=True)
        step = size - overlap

        for i in range(0, len(lines), step):
            chunk_lines = lines[i : i + size]
            chunk_text = "".join(chunk_lines)
            chunks.append(chunk_text)

            if len(chunks) >= max_chunks:
                break

    return chunks


def select(content: str, ranges: list[tuple[int, int]], max_chars: int = 10000) -> str:
    """
    Extract and combine multiple ranges from content.

    Args:
        content: The content to select from
        ranges: List of (start, end) tuples (character positions)
        max_chars: Maximum total characters to return

    Returns:
        Combined content from all ranges, separated by markers

    Examples:
        >>> select(P, [(0, 100), (500, 600)])  # First and middle sections
        >>> select(P, [(0, 50), (-50, None)])  # Beginning and end
    """
    if not ranges:
        return ""

    parts: list[str] = []
    total_chars = 0

    for i, (start, end) in enumerate(ranges):
        # Use peek for each range (handles bounds checking)
        part = peek(content, start, end, max_chars=max_chars - total_chars)

        # Check if truncated
        if total_chars + len(part) > max_chars:
            parts.append(f"... (output limit reached at range {i + 1}/{len(ranges)})")
            break

        # Add range marker
        if i > 0:
            parts.append(f"\n{'=' * 60}")
            parts.append(f"Range {i + 1}: chars {start} to {end}")
            parts.append(f"{'=' * 60}\n")

        parts.append(part)
        total_chars += len(part)

    return "".join(parts)
