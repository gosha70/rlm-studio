# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Prompt templates for RLM system prompts.

This module provides templated system prompts that can be customized
and versioned. Prompts are loaded from YAML files for easy editing and
version control.
"""

import json
from pathlib import Path
from typing import Any

# Cache for loaded templates
_prompt_version_cache: dict[str, str] = {}
_template_cache: dict[str, str] = {}
_yaml_doc_cache: dict[str, Any] = {}


def get_system_prompt_version(version: str = "2.0") -> str:
    """The ``version`` recorded inside the prompt file for ``version``.

    The file is selected by the *major* version (``2.0`` →
    ``system_prompt_v2_0.yaml``); its metadata carries the exact protocol
    revision (``2.2`` once ``read_result`` exists), which the RLM runtime
    fingerprint records so a trace says which prompt the model saw.
    """
    cache_key = f"system_prompt_v{version}"
    if cache_key not in _prompt_version_cache:
        get_default_system_prompt(version=version)
    return _prompt_version_cache.get(cache_key, version)


def get_default_system_prompt(version: str = "2.0") -> str:
    """
    Get the default system prompt template from versioned template file.

    Templates are stored as discoverable files: system_prompt_v{major}_{minor}.txt

    Args:
        version: Version of the prompt template to use (e.g., "1.0", "2.0")

    Returns:
        System prompt template string

    Raises:
        ValueError: If version is not supported
        FileNotFoundError: If template file doesn't exist

    Example:
        >>> template = get_default_system_prompt("1.0")
        >>> # Loads from system_prompt_v1_0.txt
    """
    # Check cache first
    cache_key = f"system_prompt_v{version}"
    if cache_key in _template_cache:
        return _template_cache[cache_key]

    # Convert version to filename format: "1.0" -> "v1_0"
    version_str = version.replace(".", "_")

    # All versioned prompt files use YAML — the single canonical format.
    import importlib.resources as pkg_resources

    import yaml

    pkg = pkg_resources.files("rlmstudio.prompts")
    for ext in (".yaml", ".yml"):
        filename = f"system_prompt_v{version_str}{ext}"
        try:
            raw = (pkg / filename).read_text(encoding="utf-8")
        except (FileNotFoundError, TypeError):
            continue

        data = yaml.safe_load(raw)
        if not isinstance(data, dict) or "template" not in data:
            raise ValueError(f"YAML prompt file must have a 'template' key: {filename}")
        template = str(data["template"])
        _template_cache[cache_key] = template
        _prompt_version_cache[cache_key] = str(data.get("version") or version)
        return template

    raise ValueError(
        f"Unsupported system prompt version: {version}. "
        f"No template file found (tried system_prompt_v{version_str}{{.yaml,.yml}})."
    )


def format_system_prompt(
    template: str | None = None, prompt_length: int = 0, version: str = "2.0", **kwargs: Any
) -> str:
    """
    Format a system prompt template with given parameters.

    Args:
        template: Custom template string. If None, uses default for version
        prompt_length: Length of the content being analyzed
        version: Version of default template to use if template is None
        **kwargs: Additional template variables

    Returns:
        Formatted system prompt string

    Example:
        >>> prompt = format_system_prompt(prompt_length=12345)
        >>> assert "12,345" in prompt

        >>> custom = "Content length: {prompt_length}"
        >>> prompt = format_system_prompt(template=custom, prompt_length=1000)
        >>> assert "1000" in prompt
    """
    if template is None:
        template = get_default_system_prompt(version=version)

    # Format with provided kwargs
    format_vars = {"prompt_length": prompt_length, **kwargs}

    try:
        return template.format(**format_vars)
    except KeyError as e:
        raise ValueError(f"Missing template variable: {e}") from e


def get_mode_system_prompt(mode: str, template_name: str = "Default") -> str:
    """Get the default system prompt for a given execution mode.

    Loads from ``system_prompt_templates.json`` (the named profile).

    Args:
        mode: Execution mode — ``"direct"``, ``"rag"``, or ``"rlm"``.
        template_name: Profile name in the templates JSON (default ``"Default"``).

    Returns:
        System prompt string.

    Raises:
        ValueError: If the mode or template name is not found.

    Example:
        >>> prompt = get_mode_system_prompt("direct")
        >>> assert "helpful" in prompt
    """
    cache_key = f"mode_prompt_{template_name}_{mode}"
    if cache_key in _template_cache:
        return _template_cache[cache_key]

    import importlib.resources as pkg_resources

    pkg = pkg_resources.files("rlmstudio.prompts")
    raw = (pkg / "system_prompt_templates.json").read_text(encoding="utf-8")
    data: dict[str, Any] = json.loads(raw)

    tpl = data.get(template_name)
    if tpl is None:
        raise ValueError(
            f"System prompt template '{template_name}' not found in system_prompt_templates.json"
        )
    prompt = tpl.get(mode)
    if not prompt:
        raise ValueError(f"No system prompt for mode='{mode}' in template '{template_name}'")
    _template_cache[cache_key] = str(prompt)
    return str(prompt)


def get_rlm_message(key: str) -> str:
    """Get a named operational message from ``rlm_messages.yaml``.

    Covers re-prompt messages and quality-gate feedback strings.

    Args:
        key: Message key, e.g. ``"reprompt"``, ``"quality_gate_impossibility"``,
             ``"quality_gate_api_existence"``.

    Returns:
        Message string.

    Raises:
        ValueError: If the key is not found in the YAML file.

    Example:
        >>> msg = get_rlm_message("reprompt")
        >>> assert "FINAL" in msg
    """
    if "rlm_messages" not in _yaml_doc_cache:
        import importlib.resources as pkg_resources

        import yaml

        pkg = pkg_resources.files("rlmstudio.prompts")
        raw = (pkg / "rlm_messages.yaml").read_text(encoding="utf-8")
        _yaml_doc_cache["rlm_messages"] = yaml.safe_load(raw)

    doc: dict[str, str] = _yaml_doc_cache["rlm_messages"]
    msg = doc.get(key)
    if not msg:
        raise ValueError(f"RLM message key '{key}' not found in rlm_messages.yaml")
    return str(msg).rstrip("\n")


def load_prompt_from_file(filepath: Path) -> str:
    """
    Load a prompt template from a file.

    Supports:
    - .txt files: Plain text templates
    - .md files: Markdown templates
    - .yaml/.yml files: YAML with 'template' key

    Args:
        filepath: Path to template file

    Returns:
        Template string

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Template file not found: {filepath}")

    suffix = filepath.suffix.lower()

    if suffix in [".txt", ".md"]:
        return filepath.read_text()
    elif suffix in [".yaml", ".yml"]:
        import yaml

        with open(filepath) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"YAML file must be a mapping, got {type(data).__name__}: {filepath}")
        if "template" not in data:
            raise ValueError(f"YAML file must have 'template' key: {filepath}")
        template = data["template"]
        if not isinstance(template, str):
            raise ValueError(
                f"YAML 'template' value must be a string, got {type(template).__name__}: {filepath}"
            )
        return template
    else:
        raise ValueError(f"Unsupported template file format: {suffix}")


def save_prompt_to_file(template: str, filepath: Path) -> None:
    """
    Save a prompt template to a file.

    Args:
        template: Template string to save
        filepath: Path where to save the template
    """
    suffix = filepath.suffix.lower()

    if suffix in [".txt", ".md"]:
        filepath.write_text(template)
    elif suffix in [".yaml", ".yml"]:
        import yaml

        data = {"template": template, "version": "1.0"}
        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
    else:
        raise ValueError(f"Unsupported template file format: {suffix}")
