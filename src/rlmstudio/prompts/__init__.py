# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0

"""Prompt templates for RLM."""

from .templates import (
    format_system_prompt,
    get_default_system_prompt,
    get_mode_system_prompt,
    get_rlm_message,
    get_system_prompt_version,
)

__all__ = [
    "get_default_system_prompt",
    "format_system_prompt",
    "get_mode_system_prompt",
    "get_rlm_message",
    "get_system_prompt_version",
]
