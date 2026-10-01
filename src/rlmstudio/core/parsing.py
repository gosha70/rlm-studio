# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Parsing utilities for extracting code and answers from LLM responses."""

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ParsedResponse:
    """Result of parsing an LLM response."""

    code: str | None = None
    final_answer: str | None = None
    final_var: str | None = None
    raw_text: str = ""
    is_inspect: bool = False  # True when code was generated from a v2.0 inspect action
    inspect_action: Any = None
    """The parsed v2.0 :class:`~rlmstudio.core.actions.InspectAction`, when the
    response was a JSON inspect action.  The controller executes it directly
    (``application/services/inspect_dispatcher.py``); ``code`` then holds the
    display form recorded in the trace, not code sent to the sandbox."""

    @property
    def has_code(self) -> bool:
        """Check if response contains code to execute."""
        return self.code is not None and len(self.code.strip()) > 0

    @property
    def has_final(self) -> bool:
        """Check if response contains a final answer."""
        return self.final_answer is not None or self.final_var is not None

    @property
    def is_complete(self) -> bool:
        """Check if this is a final response (has answer, no more code)."""
        return self.has_final and not self.has_code


def extract_python_code(text: str) -> str | None:
    """
    Extract Python code from markdown code blocks.

    Looks for ```python or ``` fenced code blocks and returns the code inside.
    If multiple blocks exist, returns the first one.

    Args:
        text: Text potentially containing code blocks

    Returns:
        Extracted code or None if no code block found

    Examples:
        >>> extract_python_code("```python\\nx = 1\\n```")
        'x = 1'
        >>> extract_python_code("```\\ny = 2\\n```")
        'y = 2'
        >>> extract_python_code("No code here")
        None
    """
    # Try python-specific code block first
    python_pattern = r"```python\s*\n(.*?)\n```"
    match = re.search(python_pattern, text, re.DOTALL | re.IGNORECASE)

    if match:
        return match.group(1).strip()

    # Fall back to generic code block
    generic_pattern = r"```\s*\n(.*?)\n```"
    match = re.search(generic_pattern, text, re.DOTALL)

    if match:
        code = match.group(1).strip()
        # Make sure it's not another language
        first_line = code.split("\n")[0].lower()
        if not any(
            first_line.startswith(lang)
            for lang in ["javascript", "java", "c++", "rust", "go", "sql"]
        ):
            return code

    return None


def extract_final_answer(text: str) -> str | None:
    """
    Extract final answer from FINAL: prefix.

    Looks for lines starting with "FINAL:" and returns everything after,
    including subsequent lines (for multi-line answers).
    Can be case-insensitive.

    Args:
        text: Text potentially containing final answer

    Returns:
        Final answer or None if not found

    Examples:
        >>> extract_final_answer("After analysis...\\nFINAL: The answer is 42")
        'The answer is 42'
        >>> extract_final_answer("final: Done")
        'Done'
        >>> extract_final_answer("No answer here")
        None
        >>> extract_final_answer("FINAL: Line 1\\nLine 2\\nLine 3")
        'Line 1\\nLine 2\\nLine 3'
    """
    # Match FINAL: at start of line and capture everything after to end of text
    # Uses DOTALL so .* matches newlines for multi-line answers
    pattern = r"^FINAL:\s*(.*)"
    match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE | re.DOTALL)

    if match:
        return match.group(1).strip()

    return None


def extract_final_var(text: str) -> str | None:
    """
    Extract variable name from FINAL_VAR: directive.

    Looks for lines with "FINAL_VAR: variable_name" to indicate which
    variable in the execution environment contains the final answer.

    Args:
        text: Text potentially containing FINAL_VAR directive

    Returns:
        Variable name or None if not found

    Examples:
        >>> extract_final_var("FINAL_VAR: result")
        'result'
        >>> extract_final_var("final_var: answer")
        'answer'
        >>> extract_final_var("No directive here")
        None
    """
    # Match FINAL_VAR: with variable name, case-insensitive
    pattern = r"^FINAL_VAR:\s*([a-zA-Z_][a-zA-Z0-9_]*)$"
    match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)

    if match:
        return match.group(1).strip()

    return None


def parse_response(text: str) -> ParsedResponse:
    """
    Parse LLM response to extract code blocks and final answers.

    This is the main parsing function that tries to extract:
    1. Python code blocks (for execution)
    2. FINAL: answers (direct responses)
    3. FINAL_VAR: directives (variable references)

    Args:
        text: Raw LLM response text

    Returns:
        ParsedResponse with extracted components

    Examples:
        >>> resp = parse_response("```python\\nx = 1\\n```")
        >>> resp.has_code
        True
        >>> resp.code
        'x = 1'

        >>> resp = parse_response("FINAL: Done")
        >>> resp.has_final
        True
        >>> resp.final_answer
        'Done'
    """
    final_answer = extract_final_answer(text)
    final_var = extract_final_var(text)
    # If the model declares FINAL/FINAL_VAR, treat it as terminal and ignore code.
    code = None if (final_answer or final_var) else extract_python_code(text)
    return ParsedResponse(
        code=code,
        final_answer=final_answer,
        final_var=final_var,
        raw_text=text,
    )


def format_code_for_display(code: str, max_lines: int = 10) -> str:
    """
    Format code for display in logs/traces, with truncation if needed.

    Args:
        code: Python code to format
        max_lines: Maximum number of lines to show

    Returns:
        Formatted code string, possibly truncated
    """
    lines = code.split("\n")

    if len(lines) <= max_lines:
        return code

    truncated = "\n".join(lines[:max_lines])
    remaining = len(lines) - max_lines
    return f"{truncated}\n... ({remaining} more lines)"


def format_result_for_llm(result: dict) -> str:
    """
    Format execution result for inclusion in next LLM message.

    Converts the PyReplEnv result dict into a readable format for the LLM.

    Args:
        result: Result dict from PyReplEnv.execute()

    Returns:
        Formatted string for LLM consumption
    """
    parts = []

    # Show stdout if present
    if result.get("stdout"):
        stdout = result["stdout"].strip()
        if stdout:
            parts.append(f"Output:\n{stdout}")

    # Show stderr if present
    if result.get("stderr"):
        stderr = result["stderr"].strip()
        if stderr:
            parts.append(f"Errors:\n{stderr}")

    # Show exception if present
    if result.get("exception"):
        parts.append(f"Exception:\n{result['exception']}")

    # Show timeout if occurred
    if result.get("timeout"):
        parts.append("⚠️  Execution timed out")

    # If nothing to show, indicate success
    if not parts:
        parts.append("✓ Code executed successfully (no output)")

    return "\n\n".join(parts)
