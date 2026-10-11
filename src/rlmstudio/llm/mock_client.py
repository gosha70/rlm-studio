# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0

"""Mock LLM client for testing."""


class MockLLMClient:
    """
    Mock LLM client that returns pre-programmed responses.

    Useful for testing RLM without requiring actual LLM API calls.

    Example:
        >>> client = MockLLMClient([
        ...     "```python\\nresult = peek(0, 10)\\nprint(result)\\n```",
        ...     "FINAL: The answer is test"
        ... ])
        >>> response1 = client.complete([{"role": "user", "content": "test"}])
        >>> response2 = client.complete([{"role": "user", "content": "continue"}])
    """

    def __init__(self, responses: list[str]):
        """
        Initialize mock client with pre-programmed responses.

        Args:
            responses: List of responses to return in order.
                      After exhausting the list, repeats the last response.
        """
        if not responses:
            raise ValueError("MockLLMClient requires at least one response")

        self.responses = responses
        self.call_count = 0
        self.call_history: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        """
        Return next pre-programmed response.

        Args:
            messages: Message history (recorded but not used for mock)

        Returns:
            Next response from the pre-programmed list
        """
        # Record the call (store a copy to avoid reference issues)
        self.call_history.append(list(messages))

        # Get response index (clamp to last response if exceeded)
        idx = min(self.call_count, len(self.responses) - 1)
        response = self.responses[idx]

        self.call_count += 1

        return response

    def reset(self) -> None:
        """Reset call count and history."""
        self.call_count = 0
        self.call_history = []
