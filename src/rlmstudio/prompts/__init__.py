# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

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
