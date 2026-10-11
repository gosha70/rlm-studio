# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0

"""Recursive Language Model (RLM) controller.

Based on the RLM paper (arXiv:2512.24601), this module implements the core
controller loop that:
1. Loads large prompts as variables in a REPL environment
2. Allows LLM to write Python code to explore the content
3. Executes code and feeds results back to LLM
4. Manages recursion and budgets
5. Extracts final answers
"""

import time
from dataclasses import dataclass
from typing import Any, Protocol

from ..application.sandbox_vars import (
    MODE_DIRECT,
    MODE_RLM,
    TRACE_KEY_CONTENT,
    TRACE_KEY_MODE,
    TRACE_KEY_ROLE,
    TRACE_KEY_STEP,
)
from ..config import RLMConfig
from ..envs.pyrepl_env import PyReplEnv
from ..prompts import format_system_prompt, get_mode_system_prompt, get_rlm_message
from ..tools.recursion import create_subcall
from .actions import ParseError, parse_action
from .budget import BudgetTracker, TokenUsage, estimate_tokens
from .comparison import ComparisonResult, ExecutionMetrics
from .errors import BudgetExceeded
from .parsing import ParsedResponse, format_result_for_llm, parse_response


class LLMClient(Protocol):
    """Protocol for LLM providers.

    Any LLM client must implement this interface to work with RLM.
    """

    def complete(self, messages: list[dict[str, str]]) -> str:
        """
        Generate completion from messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            Generated text response from LLM
        """
        ...


@dataclass
class RLMResult:
    """Result from RLM execution."""

    answer: str
    """Final answer extracted from LLM"""

    steps: int
    """Number of execution steps taken"""

    trace: list[dict[str, Any]]
    """Full execution trace (messages + execution results)"""

    success: bool = True
    """Whether execution completed successfully"""

    error: str | None = None
    """Error message if execution failed"""

    total_input_tokens: int = 0
    """Total input tokens across all LLM calls (from API metadata)"""

    total_output_tokens: int = 0
    """Total output tokens across all LLM calls (from API metadata)"""


class RLM:
    """
    Recursive Language Model controller.

    Implements the RLM paradigm from arXiv:2512.24601 where:
    - Large prompts are loaded as variables in a REPL environment
    - LLM writes Python code to explore the content
    - Code execution results feed back to LLM
    - Process repeats until final answer is found

    Example:
        >>> from rlmstudio import RLM, RLMConfig
        >>> from rlmstudio.llm import MockLLMClient
        >>>
        >>> client = MockLLMClient(["```python\\nx = peek(0, 10)\\n```", "FINAL: Done"])
        >>> rlm = RLM(client=client, config=RLMConfig(max_steps=10))
        >>> result = rlm.run(prompt="Large content here...", query="Summarize this")
        >>> print(result.answer)  # "Done"
    """

    def __init__(
        self,
        client: LLMClient,
        config: RLMConfig | None = None,
        budget_tracker: BudgetTracker | None = None,
    ):
        """
        Initialize RLM controller.

        Args:
            client: LLM client implementing LLMClient protocol
            config: Configuration for execution limits and security
            budget_tracker: Optional budget tracker (created if not provided)
        """
        self.client = client
        self.config = config or RLMConfig()
        self.env: PyReplEnv | None = None
        self._budget_tracker = budget_tracker  # For recursion tracking

    def run(self, prompt: str, query: str, system_prompt: str | None = None) -> RLMResult:
        """
        Run RLM on a prompt to answer a query.

        This is the main entry point for RLM execution:
        1. Initializes a REPL environment with the prompt as variable 'P'
        2. Sends query to LLM with instructions about available tools
        3. Parses LLM response for code or final answer
        4. If code: executes it, shows results to LLM, repeats
        5. If final: extracts answer and returns
        6. Enforces budget limits (max_steps)

        Args:
            prompt: Large content to analyze (the "P" variable in paper)
            query: Question to answer about the prompt
            system_prompt: Optional custom system prompt (uses default if None)

        Returns:
            RLMResult with answer, execution trace, and statistics

        Raises:
            BudgetExceeded: If max_steps is reached without final answer
        """
        # Initialize REPL environment with prompt as 'P'
        self.env = PyReplEnv(
            safe_mode=self.config.execution.default_safe_mode,
            allowed_imports=list(self.config.security.safe_modules),
            max_exec_time_s=self.config.execution.default_timeout,
            max_stdout_chars=self.config.execution.max_output_chars,
        )
        self.env.set_content(prompt)

        # Bind subcall function to REPL environment for recursion support
        subcall_func = create_subcall(self)
        self.env.env_globals["subcall"] = subcall_func

        # Build message history
        messages = []

        # Add system prompt
        if system_prompt is None:
            system_prompt = self._build_system_prompt(len(prompt))
        messages.append({"role": "system", "content": system_prompt})

        # Add user query
        messages.append({"role": "user", "content": query})

        # Main execution loop
        trace = []
        steps = 0
        last_execution_failed = False  # Track if previous execution had errors
        cumulative_input_tokens = 0
        cumulative_output_tokens = 0

        try:
            while steps < self.config.execution.max_steps:
                steps += 1

                # Get LLM response — prefer complete_with_metadata() for real token counts
                step_input_tokens = 0
                step_output_tokens = 0
                if hasattr(self.client, "complete_with_metadata"):
                    try:
                        llm_response = self.client.complete_with_metadata(messages)
                        response = llm_response.content
                        step_input_tokens = llm_response.input_tokens or 0
                        step_output_tokens = llm_response.output_tokens or 0
                    except (NotImplementedError, AttributeError):
                        response = self.client.complete(messages)
                else:
                    response = self.client.complete(messages)

                cumulative_input_tokens += step_input_tokens
                cumulative_output_tokens += step_output_tokens

                # Add to trace
                trace.append(
                    {
                        "step": steps,
                        "role": "assistant",
                        "content": response,
                        "input_tokens": step_input_tokens,
                        "output_tokens": step_output_tokens,
                    }
                )

                # Parse response - try JSON format (v2.0) first, fall back to markdown (v1.0)
                parsed = self._parse_rlm_response(response)

                # Check if complete - FINAL is a HARD STOP
                # CRITICAL: According to control-loop engineering analysis,
                # FINAL must be a terminal state. Once detected, execution stops immediately.
                # No retries, no rejections, no further loop iterations.
                # This prevents: apology loops, multiple FINAL attempts, and hedging behavior.
                if parsed.is_complete:
                    # Log warning if there was an unresolved execution failure
                    # But still accept the FINAL answer (model may have used direct reasoning)
                    if last_execution_failed:
                        trace.append(
                            {
                                "step": steps,
                                "role": "system",
                                "content": "⚠️ Warning: FINAL provided after execution failure. "
                                "Model may have used direct reasoning instead of code execution.",
                            }
                        )

                    # Extract and return final answer - NO FURTHER PROCESSING
                    answer = self._extract_final_answer(parsed)

                    return RLMResult(
                        answer=answer,
                        steps=steps,
                        trace=trace,
                        success=True,
                        total_input_tokens=cumulative_input_tokens,
                        total_output_tokens=cumulative_output_tokens,
                    )

                # Execute code if present
                if parsed.has_code and parsed.code is not None:
                    # Execute in REPL
                    result = self.env.execute(parsed.code)

                    # Track if execution failed or succeeded
                    # This flag persists until successful execution or FINAL is provided
                    last_execution_failed = result.get("exception") is not None or result.get(
                        "timeout", False
                    )

                    # Format for LLM
                    formatted_result = format_result_for_llm(result)

                    # Add to trace
                    trace.append(
                        {
                            "step": steps,
                            "role": "execution",
                            "content": formatted_result,
                            "raw_result": result,
                        }
                    )

                    # Add to message history
                    messages.append({"role": "assistant", "content": response})
                    messages.append(
                        {"role": "user", "content": f"Execution result:\n{formatted_result}"}
                    )
                else:
                    # No code and no FINAL - unclear response
                    # Prompt the model to provide executable code or a final answer
                    # Note: last_execution_failed flag persists until successful execution
                    # or FINAL is provided (where it's logged as a warning)

                    messages.append({"role": "assistant", "content": response})
                    messages.append(
                        {
                            "role": "user",
                            "content": get_rlm_message("reprompt"),
                        }
                    )

            # Max steps exceeded
            raise BudgetExceeded(
                f"Maximum steps ({self.config.execution.max_steps}) exceeded without finding final answer"
            )

        except BudgetExceeded:
            raise
        except Exception as e:
            # Other errors
            return RLMResult(
                answer="",
                steps=steps,
                trace=trace,
                success=False,
                error=str(e),
                total_input_tokens=cumulative_input_tokens,
                total_output_tokens=cumulative_output_tokens,
            )

    def _extract_final_answer(self, parsed: ParsedResponse) -> str:
        """
        Extract final answer from parsed response.

        Handles both FINAL: direct answers and FINAL_VAR: variable references.

        Args:
            parsed: Parsed LLM response

        Returns:
            Final answer string
        """
        if parsed.final_answer:
            return parsed.final_answer

        if parsed.final_var and self.env:
            # Get variable from environment
            var_value = self.env.get_var(parsed.final_var)
            if var_value is not None:
                return str(var_value)
            else:
                return f"Error: Variable '{parsed.final_var}' not found in environment"

        return "Error: No final answer found"

    def _parse_rlm_response(self, response: str) -> ParsedResponse:
        """
        Parse RLM response in either JSON (v2.0) or markdown (v1.0) format.

        Tries JSON format first (for v2.0 prompts), falls back to markdown parsing (for v1.0).
        Converts JSON actions to ParsedResponse format for backward compatibility.

        Args:
            response: Raw LLM response text

        Returns:
            ParsedResponse object
        """
        # Try JSON format first (v2.0)
        try:
            action_type, action_obj = parse_action(response)

            # Convert JSON action to ParsedResponse
            if action_type == "final":
                return ParsedResponse(final_answer=action_obj.answer, raw_text=response)
            elif action_type == "inspect":
                code = self._inspect_to_code(action_obj)
                if code:
                    return ParsedResponse(code=code, raw_text=response)
                return ParsedResponse(raw_text=response)
            elif action_type == "subcall":
                # Execute subcall via the bound function
                subcall_func = self.env.env_globals.get("subcall") if self.env else None
                if subcall_func and hasattr(action_obj, "query") and hasattr(action_obj, "prompt"):
                    # Generate code representation for tracing
                    # Note: SubcallAction uses 'prompt' but subcall() function uses 'content'
                    code = f"subcall(content={repr(action_obj.prompt)}, query={repr(action_obj.query)})"

                    # Actually execute the subcall
                    try:
                        # Map action_obj.prompt to content parameter
                        sub_answer = subcall_func(content=action_obj.prompt, query=action_obj.query)
                        # Return as if it was code execution with result
                        return ParsedResponse(
                            code=code, final_answer=str(sub_answer), raw_text=response
                        )
                    except Exception as e:
                        # Return error as execution result
                        return ParsedResponse(
                            code=code, final_answer=f"Error in subcall: {str(e)}", raw_text=response
                        )
                else:
                    # Subcall function not available or malformed action
                    return ParsedResponse(raw_text=response)

        except (ParseError, Exception):
            # Retry: extract just the balanced JSON, ignoring trailing prose.
            try:
                from rlmstudio.core.actions import _extract_balanced_json

                json_only = _extract_balanced_json(response)
                if json_only:
                    action_type, action_obj = parse_action(json_only)
                    if action_type == "final":
                        return ParsedResponse(final_answer=action_obj.answer, raw_text=response)
                    elif action_type == "inspect":
                        code = self._inspect_to_code(action_obj)
                        if code:
                            return ParsedResponse(code=code, raw_text=response)
                    elif action_type == "subcall":
                        subcall_func = self.env.env_globals.get("subcall") if self.env else None
                        if (
                            subcall_func
                            and hasattr(action_obj, "query")
                            and hasattr(action_obj, "prompt")
                        ):
                            code = f"subcall(content={repr(action_obj.prompt)}, query={repr(action_obj.query)})"
                            try:
                                sub_answer = subcall_func(
                                    content=action_obj.prompt, query=action_obj.query
                                )
                                return ParsedResponse(
                                    code=code, final_answer=str(sub_answer), raw_text=response
                                )
                            except Exception as e:
                                return ParsedResponse(
                                    code=code,
                                    final_answer=f"Error in subcall: {e}",
                                    raw_text=response,
                                )
            except Exception:
                pass

        # Fall back to markdown parsing (v1.0)
        return parse_response(response)

    @staticmethod
    def _inspect_to_code(action_obj: Any) -> str | None:
        """Convert a JSON inspect action to executable Python code."""
        tool: str = action_obj.tool
        args: dict[str, Any] = action_obj.args
        if tool == "grep":
            return (
                f"print(grep("
                f"pattern={repr(args.get('pattern'))}, "
                f"context_lines={args.get('context_lines', 2)}, "
                f"max_matches={args.get('max_matches', 100)}, "
                f"ignore_case={args.get('ignore_case', False)}, "
                f"use_regex={args.get('use_regex', False)}))"
            )
        elif tool == "peek":
            return (
                f"print(peek("
                f"start={args.get('start', 0)}, "
                f"end={args.get('end')}, "
                f"max_chars={args.get('max_chars', 10000)}))"
            )
        elif tool == "peek_file":
            return (
                f"print(peek_file("
                f"file_no={args.get('file_no')}, "
                f"start={args.get('start', 0)}, "
                f"end={args.get('end')}, "
                f"max_chars={args.get('max_chars', 10000)}))"
            )
        elif tool == "grep_file":
            return (
                f"print(grep_file("
                f"file_no={args.get('file_no')}, "
                f"pattern={repr(args.get('pattern'))}, "
                f"context_lines={args.get('context_lines', 2)}, "
                f"max_matches={args.get('max_matches', 100)}, "
                f"ignore_case={args.get('ignore_case', False)}, "
                f"use_regex={args.get('use_regex', False)}))"
            )
        elif tool == "outline_file":
            return (
                f"print(outline_file("
                f"file_no={args.get('file_no')}, "
                f"max_lines={args.get('max_lines', 40)}, "
                f"max_chars={args.get('max_chars', 8000)}))"
            )
        elif tool == "select":
            return f"print(select(ranges={args.get('ranges')}))"
        elif tool == "chunk":
            return (
                f"print(chunk("
                f"size={args.get('size', 1000)}, "
                f"overlap={args.get('overlap', 0)}, "
                f"by={repr(args.get('by', 'chars'))}, "
                f"max_chunks={args.get('max_chunks', 100)}))"
            )
        return None

    def _build_system_prompt(self, prompt_length: int) -> str:
        """
        Build default system prompt for RLM.

        Uses the templated prompt system for easy customization and versioning.

        Args:
            prompt_length: Length of the prompt content in characters

        Returns:
            System prompt string
        """
        return format_system_prompt(prompt_length=prompt_length)

    def run_direct(self, prompt: str, query: str, system_prompt: str | None = None) -> RLMResult:
        """
        Run in direct mode (no RLM exploration, just send full prompt to LLM).

        This mode sends the entire prompt and query to the LLM in a single request,
        bypassing the RLM exploration loop. Useful for comparison and baseline testing.

        Args:
            prompt: Large content to analyze
            query: Question to answer about the prompt
            system_prompt: Optional custom system prompt

        Returns:
            RLMResult with answer and metrics
        """
        start_time = time.time()

        # Build message for direct query
        messages = []

        if system_prompt is None:
            system_prompt = get_mode_system_prompt(MODE_DIRECT)

        messages.append({"role": "system", "content": system_prompt})

        # Combine prompt and query
        combined_content = f"Content:\n{prompt}\n\nQuestion: {query}"

        messages.append({"role": "user", "content": combined_content})

        try:
            # Get LLM response
            response = self.client.complete(messages)

            _elapsed_time = time.time() - start_time

            return RLMResult(
                answer=response,
                steps=0,  # Direct mode = 0 steps
                trace=[
                    {
                        TRACE_KEY_STEP: 0,
                        TRACE_KEY_ROLE: "assistant",
                        TRACE_KEY_CONTENT: response,
                        TRACE_KEY_MODE: MODE_DIRECT,
                    }
                ],
                success=True,
            )
        except Exception as e:
            return RLMResult(answer="", steps=0, trace=[], success=False, error=str(e))

    def run_comparison(
        self, prompt: str, query: str, system_prompt: str | None = None
    ) -> ComparisonResult:
        """
        Run both RLM and Direct modes and compare results.

        This method executes the same query using both RLM exploration mode
        and direct mode, tracking metrics for comparison. Useful for demonstrating
        the value of RLM and understanding when it helps.

        Args:
            prompt: Large content to analyze
            query: Question to answer about the prompt
            system_prompt: Optional custom system prompt (used for RLM mode)

        Returns:
            ComparisonResult with metrics from both modes
        """
        comparison = ComparisonResult()

        # Run RLM mode
        if self.config.execution.enable_rlm:
            rlm_start = time.time()
            try:
                rlm_result = self.run(prompt, query, system_prompt)
                rlm_elapsed = time.time() - rlm_start

                # Calculate token usage for RLM mode
                rlm_tokens = TokenUsage()
                for trace_item in rlm_result.trace:
                    if trace_item.get("role") == "assistant":
                        rlm_tokens.add_output(estimate_tokens(trace_item["content"]))
                    elif trace_item.get("role") == "execution":
                        # Count execution results as input to next step
                        rlm_tokens.add_input(estimate_tokens(trace_item["content"]))

                # Add initial prompt
                rlm_tokens.add_input(estimate_tokens(prompt + query))

                comparison.rlm_metrics = ExecutionMetrics(
                    mode=MODE_RLM,
                    answer=rlm_result.answer,
                    steps=rlm_result.steps,
                    tokens=rlm_tokens,
                    elapsed_time=rlm_elapsed,
                    success=rlm_result.success,
                    error=rlm_result.error,
                    trace=rlm_result.trace,
                )
            except Exception as e:
                rlm_elapsed = time.time() - rlm_start
                comparison.rlm_metrics = ExecutionMetrics(
                    mode=MODE_RLM,
                    answer="",
                    steps=0,
                    tokens=TokenUsage(),
                    elapsed_time=rlm_elapsed,
                    success=False,
                    error=str(e),
                    trace=[],
                )

        # Run Direct mode
        direct_start = time.time()
        try:
            direct_result = self.run_direct(prompt, query)
            direct_elapsed = time.time() - direct_start

            # Calculate token usage for Direct mode
            direct_tokens = TokenUsage()
            direct_tokens.add_input(estimate_tokens(prompt + query))
            direct_tokens.add_output(estimate_tokens(direct_result.answer))

            comparison.direct_metrics = ExecutionMetrics(
                mode=MODE_DIRECT,
                answer=direct_result.answer,
                steps=0,
                tokens=direct_tokens,
                elapsed_time=direct_elapsed,
                success=direct_result.success,
                error=direct_result.error,
                trace=direct_result.trace,
            )
        except Exception as e:
            direct_elapsed = time.time() - direct_start
            comparison.direct_metrics = ExecutionMetrics(
                mode=MODE_DIRECT,
                answer="",
                steps=0,
                tokens=TokenUsage(),
                elapsed_time=direct_elapsed,
                success=False,
                error=str(e),
                trace=[],
            )

        return comparison
