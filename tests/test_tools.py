"""Tests for content navigation tools."""

import re
import time

import pytest

from rlmstudio.envs import PyReplEnv
from rlmstudio.tools import PatternTimeoutError, grep, grep_file

# Sample content for testing
SAMPLE_TEXT = """Line 1: Introduction to the document
Line 2: This is a test file
Line 3: It contains multiple lines
Line 4: With various patterns
Line 5: ERROR: Something went wrong
Line 6: WARNING: Check this
Line 7: INFO: All systems operational
Line 8: ERROR: Another error occurred
Line 9: Final line of the document
Line 10: End of file"""

LONG_TEXT = "x" * 50000  # 50K chars for truncation tests


def _build_multi_file_text() -> str:
    """Create a realistic multi-file payload with a document index."""
    records = [
        (
            "kg_book.txt",
            "Knowledge Graphs Applied\nbrief contents\n1 Knowledge graphs and LLMs\n2 Intelligent systems\n",
        ),
        (
            "search_book.txt",
            "AI-Powered Search\nbrief contents\n1 Introducing AI-powered search\n2 Semantic search with knowledge graphs\n",
        ),
    ]
    separator = "\n\n---\n\n"
    section_headers = [f"[File {i + 1}: {name}]\n\n" for i, (name, _body) in enumerate(records)]
    sections = [f"{section_headers[i]}{records[i][1]}" for i in range(len(records))]
    offsets = [(0, 0, 0) for _ in records]

    for _ in range(8):
        index_lines = [
            f"[DOCUMENT INDEX — {len(records)} files attached]",
            "Read this index first with peek(0, 1200).",
            "Each file entry includes exact character offsets within P:",
        ]
        for i, ((name, _body), (file_start, content_start, file_end_exclusive)) in enumerate(
            zip(records, offsets, strict=True), start=1
        ):
            index_lines.append(
                f'  {i}. "{name}" (file_start={file_start}, content_start={content_start}, '
                f"file_end_exclusive={file_end_exclusive})"
            )
        index_lines.append("[END DOCUMENT INDEX]")
        index_block = "\n".join(index_lines)

        cursor = len(index_block) + 2
        new_offsets: list[tuple[int, int, int]] = []
        for i, section_text in enumerate(sections):
            file_start = cursor
            content_start = cursor + len(section_headers[i])
            file_end_exclusive = cursor + len(section_text)
            new_offsets.append((file_start, content_start, file_end_exclusive))
            cursor = file_end_exclusive
            if i < len(sections) - 1:
                cursor += len(separator)

        if new_offsets == offsets:
            break
        offsets = new_offsets

    return index_block + "\n\n" + separator.join(sections)


MULTI_FILE_TEXT = _build_multi_file_text()


def _file_2_content_start() -> int:
    match = re.search(
        r'2\. "search_book\.txt" \(file_start=\d+, content_start=(\d+), ', MULTI_FILE_TEXT
    )
    assert match is not None
    return int(match.group(1))


class TestPeekTool:
    """Test peek() content viewing tool."""

    def test_peek_basic(self):
        """Test basic peek functionality."""
        env = PyReplEnv()
        env.set_content(SAMPLE_TEXT)

        result = env.execute("print(peek(0, 10))")
        assert result["exception"] is None
        assert "Line 1:" in result["stdout"]

    def test_peek_middle_section(self):
        """Test peeking middle section."""
        env = PyReplEnv()
        env.set_content(SAMPLE_TEXT)

        result = env.execute("text = peek( 20, 50); print(len(text))")
        assert result["exception"] is None
        assert "30" in result["stdout"]  # 50-20 = 30 chars

    def test_peek_to_end(self):
        """Test peek to end when end=None."""
        env = PyReplEnv()
        env.set_content("12345")

        result = env.execute("print(peek( 2))")  # From char 2 to end
        assert result["exception"] is None
        assert "345" in result["stdout"]

    def test_peek_negative_index(self):
        """Test peek with negative indexing."""
        env = PyReplEnv()
        env.set_content("0123456789")

        result = env.execute("print(peek( -5))")  # Last 5 chars
        assert result["exception"] is None
        assert "56789" in result["stdout"]

    def test_peek_truncation(self):
        """Test peek truncates large output."""
        env = PyReplEnv()
        env.set_content(LONG_TEXT)

        result = env.execute("text = peek( 0, 50000, max_chars=100); print('done')")
        assert result["exception"] is None
        assert "done" in result["stdout"]


class TestGrepTool:
    """Test grep() search tool."""

    def test_grep_basic(self):
        """Test basic grep functionality."""
        env = PyReplEnv()
        env.set_content(SAMPLE_TEXT)

        result = env.execute("print(grep( 'ERROR'))")
        assert result["exception"] is None
        assert "Found 2 match(es)" in result["stdout"]
        assert "Line 5" in result["stdout"]
        assert "Line 8" in result["stdout"]
        assert (
            f"char_offset {SAMPLE_TEXT.index('Line 5: ERROR: Something went wrong')}"
            in result["stdout"]
        )
        assert (
            f"char_offset {SAMPLE_TEXT.index('Line 8: ERROR: Another error occurred')}"
            in result["stdout"]
        )

    def test_grep_no_matches(self):
        """Test grep when pattern not found."""
        env = PyReplEnv()
        env.set_content(SAMPLE_TEXT)

        result = env.execute("print(grep( 'NOTFOUND'))")
        assert result["exception"] is None
        assert "No matches found" in result["stdout"]

    def test_grep_with_context(self):
        """Test grep with context lines."""
        env = PyReplEnv()
        env.set_content(SAMPLE_TEXT)

        result = env.execute("print(grep( 'ERROR', context_lines=1))")
        assert result["exception"] is None
        # Should show line before and after
        assert "Line 4" in result["stdout"]  # Before Line 5
        assert "Line 6" in result["stdout"]  # After Line 5

    def test_grep_case_insensitive(self):
        """Test case-insensitive grep."""
        env = PyReplEnv()
        env.set_content(SAMPLE_TEXT)

        result = env.execute("print(grep( 'error', ignore_case=True))")
        assert result["exception"] is None
        assert "Found 2 match(es)" in result["stdout"]

    def test_grep_with_regex(self):
        """Test grep with regex pattern."""
        env = PyReplEnv()
        env.set_content(SAMPLE_TEXT)

        # Find lines with "Line" followed by digits
        result = env.execute("print(grep( r'Line \\d+', use_regex=True))")
        assert result["exception"] is None
        assert "Found" in result["stdout"]

    def test_grep_max_matches(self):
        """Test grep max_matches limit."""
        env = PyReplEnv()
        env.set_content(SAMPLE_TEXT)

        result = env.execute("print(grep( 'Line', max_matches=3))")
        assert result["exception"] is None
        assert "Found 3 match(es)" in result["stdout"]


class TestChunkTool:
    """Test chunk() splitting tool."""

    def test_chunk_by_chars(self):
        """Test chunking by characters."""
        env = PyReplEnv()
        env.set_content("0123456789")

        result = env.execute("chunks = chunk( size=3, by='chars'); print(len(chunks))")
        assert result["exception"] is None
        assert "4" in result["stdout"]  # 10 chars / 3 = 4 chunks (rounded up)

    def test_chunk_by_lines(self):
        """Test chunking by lines."""
        env = PyReplEnv()
        env.set_content(SAMPLE_TEXT)

        result = env.execute("chunks = chunk( size=3, by='lines'); print(len(chunks))")
        assert result["exception"] is None
        # SAMPLE_TEXT has 10 lines, so 3-line chunks = 4 chunks
        assert "4" in result["stdout"]

    def test_chunk_with_overlap(self):
        """Test chunking with overlap."""
        env = PyReplEnv()
        env.set_content("0123456789")

        result = env.execute("""
chunks = chunk( size=5, overlap=2, by='chars')
print(f'chunks: {len(chunks)}')
print(f'first: {chunks[0]}')
print(f'second: {chunks[1]}')
""")
        assert result["exception"] is None
        assert "01234" in result["stdout"]  # First chunk
        assert "34567" in result["stdout"]  # Second chunk (overlap of 2)

    def test_chunk_max_chunks(self):
        """Test chunk max_chunks limit."""
        env = PyReplEnv()
        env.set_content("x" * 1000)

        result = env.execute("chunks = chunk( size=10, max_chunks=5); print(len(chunks))")
        assert result["exception"] is None
        assert "5" in result["stdout"]

    def test_chunk_invalid_params(self):
        """Test chunk with invalid parameters."""
        env = PyReplEnv()
        env.set_content("test")

        # Invalid 'by' parameter
        result = env.execute("chunks = chunk( by='invalid')")
        assert result["exception"] is not None
        assert "ValueError" in result["exception"]


class TestFileScopedTools:
    """Test file-relative helpers for multi-document payloads."""

    def test_peek_file_reads_relative_to_selected_file(self):
        env = PyReplEnv()
        env.set_content(MULTI_FILE_TEXT)

        result = env.execute("print(peek_file(2, 0, 40))")
        assert result["exception"] is None
        assert "AI-Powered Search" in result["stdout"]

    def test_peek_file_normalizes_content_start_copied_from_index(self):
        env = PyReplEnv()
        env.set_content(MULTI_FILE_TEXT)
        content_start = _file_2_content_start()

        result = env.execute(f"print(peek_file(2, {content_start}, {content_start + 40}))")
        assert result["exception"] is None
        assert "AI-Powered Search" in result["stdout"]

    def test_peek_file_normalizes_global_span_when_start_is_content_start(self):
        env = PyReplEnv()
        env.set_content(MULTI_FILE_TEXT)
        content_start = _file_2_content_start()

        result = env.execute(f"print(peek_file(2, {content_start}, {content_start + 47}))")
        assert result["exception"] is None
        assert "AI-Powered Search" in result["stdout"]

    def test_grep_file_returns_file_relative_offsets(self):
        env = PyReplEnv()
        env.set_content(MULTI_FILE_TEXT)

        result = env.execute("print(grep_file(2, 'Semantic search'))")
        assert result["exception"] is None
        assert "relative to this file" in result["stdout"]
        assert "char_offset" in result["stdout"]

    def test_outline_file_extracts_document_structure(self):
        env = PyReplEnv()
        env.set_content(MULTI_FILE_TEXT)

        result = env.execute("print(outline_file(1, max_lines=5))")
        assert result["exception"] is None
        assert "Knowledge Graphs Applied" in result["stdout"]
        assert "brief contents" in result["stdout"].lower()
        assert "Knowledge graphs and LLMs" in result["stdout"]


class TestSelectTool:
    """Test select() range extraction tool."""

    def test_select_single_range(self):
        """Test selecting a single range."""
        env = PyReplEnv()
        env.set_content("0123456789")

        result = env.execute("print(select( [(2, 5)]))")
        assert result["exception"] is None
        assert "234" in result["stdout"]

    def test_select_multiple_ranges(self):
        """Test selecting multiple ranges."""
        env = PyReplEnv()
        env.set_content("0123456789")

        result = env.execute("print(select( [(0, 3), (7, 10)]))")
        assert result["exception"] is None
        assert "012" in result["stdout"]
        assert "789" in result["stdout"]
        assert "Range 2" in result["stdout"]  # Range marker

    def test_select_empty_ranges(self):
        """Test select with empty ranges list."""
        env = PyReplEnv()
        env.set_content("test")

        result = env.execute("text = select( []); print(f'len={len(text)}')")
        assert result["exception"] is None
        assert "len=0" in result["stdout"]

    def test_select_with_negative_index(self):
        """Test select with negative indices."""
        env = PyReplEnv()
        env.set_content("0123456789")

        result = env.execute("print(select( [(-5, None)]))")  # Last 5 chars
        assert result["exception"] is None
        assert "56789" in result["stdout"]

    def test_select_max_chars_limit(self):
        """Test select respects max_chars limit."""
        env = PyReplEnv()
        env.set_content("x" * 1000)

        result = env.execute("text = select( [(0, 1000)], max_chars=100); print('done')")
        assert result["exception"] is None
        assert "done" in result["stdout"]


class TestToolsIntegration:
    """Test tools working together with P variable."""

    def test_tools_available_in_env(self):
        """Test that all tools are available in environment."""
        env = PyReplEnv()

        result = env.execute(
            """
print(callable(peek))
print(callable(peek_file))
print(callable(grep))
print(callable(grep_file))
print(callable(outline_file))
print(callable(chunk))
print(callable(select))
"""
        )
        assert result["exception"] is None
        assert result["stdout"].count("True") == 7

    def test_tools_with_p_variable(self):
        """Test tools work with P variable."""
        env = PyReplEnv()
        env.set_content("Test content for P variable")

        result = env.execute("print(peek( 0, 4))")
        assert result["exception"] is None
        assert "Test" in result["stdout"]

    def test_tools_persist_after_reset(self):
        """Test tools available after reset."""
        env = PyReplEnv()
        env.execute("x = 1")
        env.reset()

        result = env.execute("print(callable(peek))")
        assert result["exception"] is None
        assert "True" in result["stdout"]

    def test_combined_tool_usage(self):
        """Test using multiple tools in sequence."""
        env = PyReplEnv()
        env.set_content(SAMPLE_TEXT)

        result = env.execute("""
# Find ERROR lines
errors = grep( 'ERROR')
print('Found errors')

# Chunk the content
chunks = chunk( size=100, by='chars')
print(f'Created {len(chunks)} chunks')

# Peek at beginning
beginning = peek( 0, 50)
print(f'Beginning length: {len(beginning)}')
""")
        assert result["exception"] is None
        assert "Found errors" in result["stdout"]
        assert "chunks" in result["stdout"]
        assert "Beginning length" in result["stdout"]


class TestToolsWithSafeMode:
    """Test tools work in safe mode."""

    def test_tools_in_safe_mode(self):
        """Test that tools work with safe mode enabled."""
        env = PyReplEnv(safe_mode=True)
        env.set_content("Safe mode test")

        result = env.execute("print(peek( 0, 4))")
        assert result["exception"] is None
        assert "Safe" in result["stdout"]

    def test_tools_with_imports_in_safe_mode(self):
        """Test tools with safe imports."""
        env = PyReplEnv(safe_mode=True)
        env.set_content("test")

        result = env.execute("""
import re
# grep uses re internally, should work
result = grep( 'test')
print('grep works')
""")
        assert result["exception"] is None
        assert "grep works" in result["stdout"]


class TestPatternTimeout:
    """``timeout`` bounds the regex engine for a model-written pattern.

    The controller runs these tools in its own process now, where nothing
    else stops a nested quantifier from backtracking for hours; `re` holds
    the GIL while it does, so the bound has to come from the engine.
    """

    CATASTROPHIC = r"(a|aa)+$"
    BUDGET = 0.5

    def test_grep_raises_once_the_budget_is_gone(self):
        text = "fine line\n" + "a" * 60 + "b\n"

        start = time.monotonic()
        with pytest.raises(PatternTimeoutError) as excinfo:
            grep(text, self.CATASTROPHIC, use_regex=True, timeout=self.BUDGET)

        assert excinfo.value.budget == self.BUDGET
        assert time.monotonic() - start < self.BUDGET * 10

    def test_grep_file_forwards_the_budget(self):
        # Length-preserving: the document index carries byte offsets, so a
        # replacement of a different length would truncate the file's slice —
        # and a run of "a" with its trailing "b" cut off matches instantly.
        victim = "2 Semantic search with knowledge graphs"
        run = "a" * (len(victim) - 1) + "b"
        content = _build_multi_file_text().replace(victim, run)

        with pytest.raises(PatternTimeoutError):
            grep_file(content, 2, self.CATASTROPHIC, use_regex=True, timeout=self.BUDGET)

    def test_without_a_budget_the_stdlib_engine_is_used(self):
        """The sandbox path keeps `re`, which its own timeout already bounds."""
        result = grep(SAMPLE_TEXT, r"ERROR", use_regex=True)

        assert "ERROR" in result

    def test_a_bounded_search_finds_the_same_matches(self):
        bounded = grep(SAMPLE_TEXT, r"ERROR:\s+\w+", use_regex=True, timeout=5.0)
        unbounded = grep(SAMPLE_TEXT, r"ERROR:\s+\w+", use_regex=True)

        assert bounded == unbounded

    def test_a_bounded_literal_search_finds_the_same_matches(self):
        bounded = grep(SAMPLE_TEXT, "ERROR", timeout=5.0)
        unbounded = grep(SAMPLE_TEXT, "ERROR")

        assert bounded == unbounded
