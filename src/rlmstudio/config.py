# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Configuration management for RLMKit."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rlmstudio.application.sandbox_vars import (
    RLM_DEFAULT_MAX_REGISTRY_BYTES,
    RLM_DEFAULT_MAX_SPILL_BYTES,
    RLM_DEFAULT_RESULT_PREVIEW_CHARS,
    RLM_DEFAULT_SPILL_RESULT_ABOVE_BYTES,
)
from rlmstudio.branding import (
    CONFIG_FILE_STEM,
    DIST_NAME,
    LEGACY_CONFIG_FILE_STEMS,
    STATE_DIR_NAME,
    env,
)

from .llm.config import LLMConfig


@dataclass
class SecurityConfig:
    """Security configuration for sandbox execution."""

    # Module-level security
    blocked_modules: set[str] = field(
        default_factory=lambda: {
            "os",
            "sys",
            "subprocess",
            "socket",
            "socketserver",
            "http",
            "urllib",
            "ftplib",
            "telnetlib",
            "smtplib",
            "pathlib",
            "shutil",
            "tempfile",
            "glob",
            "fnmatch",
            "pty",
            "tty",
            "pipes",
            "resource",
            "syslog",
            "ctypes",
            "cffi",
            "mmap",
            "signal",
            "fcntl",
            "pickle",
            "shelve",
            "dbm",
            "sqlite3",
            "importlib",
            "pkgutil",
            "modulefinder",
            "runpy",
            "__builtin__",
            "__builtins__",
            "builtins",
        }
    )

    safe_modules: set[str] = field(
        default_factory=lambda: {
            "json",
            "re",
            "math",
            "datetime",
            "time",
            "calendar",
            "decimal",
            "fractions",
            "statistics",
            "random",
            "string",
            "textwrap",
            "unicodedata",
            "itertools",
            "functools",
            "operator",
            "collections",
            "heapq",
            "bisect",
            "array",
            "copy",
            "pprint",
            "enum",
            "dataclasses",
            "typing",
        }
    )

    # Builtin-level security
    blocked_builtins: set[str] = field(
        default_factory=lambda: {
            "open",
            "input",
            "compile",
            "eval",
            "exec",
            "__import__",
            "breakpoint",
            "exit",
            "quit",
            "help",
        }
    )

    safe_builtins: set[str] = field(
        default_factory=lambda: {
            "abs",
            "all",
            "any",
            "ascii",
            "bin",
            "bool",
            "bytearray",
            "bytes",
            "callable",
            "chr",
            "classmethod",
            "complex",
            "delattr",
            "dict",
            "dir",
            "divmod",
            "enumerate",
            "filter",
            "float",
            "format",
            "frozenset",
            "getattr",
            "hasattr",
            "hash",
            "hex",
            "id",
            "int",
            "isinstance",
            "issubclass",
            "iter",
            "len",
            "list",
            "locals",
            "map",
            "max",
            "min",
            "next",
            "object",
            "oct",
            "ord",
            "pow",
            "print",
            "property",
            "range",
            "repr",
            "reversed",
            "round",
            "set",
            "setattr",
            "slice",
            "sorted",
            "staticmethod",
            "str",
            "sum",
            "super",
            "tuple",
            "type",
            "vars",
            "zip",
            "Exception",
            "ValueError",
            "TypeError",
            "KeyError",
            "IndexError",
            "AttributeError",
            "RuntimeError",
            "StopIteration",
            "AssertionError",
            "True",
            "False",
            "None",
            "NotImplemented",
            "Ellipsis",
            "__name__",
            "__doc__",
            "__package__",
            "__loader__",
            "__spec__",
        }
    )

    def add_safe_module(self, module: str) -> None:
        """Add a module to the safe list."""
        self.safe_modules.add(module)
        self.blocked_modules.discard(module)  # Remove from blocked if present

    def remove_safe_module(self, module: str) -> None:
        """Remove a module from the safe list."""
        self.safe_modules.discard(module)

    def add_blocked_module(self, module: str) -> None:
        """Add a module to the blocked list."""
        self.blocked_modules.add(module)
        self.safe_modules.discard(module)  # Remove from safe if present

    def remove_blocked_module(self, module: str) -> None:
        """Remove a module from the blocked list."""
        self.blocked_modules.discard(module)

    def is_module_allowed(self, module: str) -> bool:
        """Check if a module is allowed."""
        base_module = module.split(".")[0]
        if base_module in self.blocked_modules:
            return False
        return base_module in self.safe_modules

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "blocked_modules": sorted(self.blocked_modules),
            "safe_modules": sorted(self.safe_modules),
            "blocked_builtins": sorted(self.blocked_builtins),
            "safe_builtins": sorted(self.safe_builtins),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SecurityConfig":
        """Create config from dictionary."""
        return cls(
            blocked_modules=set(data.get("blocked_modules", [])),
            safe_modules=set(data.get("safe_modules", [])),
            blocked_builtins=set(data.get("blocked_builtins", [])),
            safe_builtins=set(data.get("safe_builtins", [])),
        )


@dataclass
class ExecutionConfig:
    """Execution environment configuration."""

    default_timeout: float = 5.0
    max_output_chars: int = 10000
    default_safe_mode: bool = True
    max_steps: int = 16  # Maximum execution steps for RLM loop
    enable_rlm: bool = True  # Enable RLM exploration mode (vs direct LLM queries)
    track_comparison_metrics: bool = True  # Track metrics for mode comparison
    use_json_protocol: bool = True  # Use JSON action protocol (v2.0) instead of markdown parsing
    max_parse_retries: int = 2  # Maximum retries for JSON parse errors
    # Result registry (specs/rlm-working-state): every v2 inspect result is
    # kept in full on the controller side; the model sees a bounded preview
    # and reads the rest through ``read_result``.
    result_preview_chars: int = RLM_DEFAULT_RESULT_PREVIEW_CHARS
    spill_result_above_bytes: int = RLM_DEFAULT_SPILL_RESULT_ABOVE_BYTES
    max_registry_bytes: int = RLM_DEFAULT_MAX_REGISTRY_BYTES
    max_spill_bytes: int = RLM_DEFAULT_MAX_SPILL_BYTES

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "default_timeout": self.default_timeout,
            "max_output_chars": self.max_output_chars,
            "default_safe_mode": self.default_safe_mode,
            "max_steps": self.max_steps,
            "enable_rlm": self.enable_rlm,
            "track_comparison_metrics": self.track_comparison_metrics,
            "result_preview_chars": self.result_preview_chars,
            "spill_result_above_bytes": self.spill_result_above_bytes,
            "max_registry_bytes": self.max_registry_bytes,
            "max_spill_bytes": self.max_spill_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionConfig":
        """Create config from dictionary."""
        return cls(
            default_timeout=data.get("default_timeout", 5.0),
            max_output_chars=data.get("max_output_chars", 10000),
            default_safe_mode=data.get("default_safe_mode", True),
            max_steps=data.get("max_steps", 16),
            enable_rlm=data.get("enable_rlm", True),
            track_comparison_metrics=data.get("track_comparison_metrics", True),
            result_preview_chars=int(
                data.get("result_preview_chars", RLM_DEFAULT_RESULT_PREVIEW_CHARS)
            ),
            spill_result_above_bytes=int(
                data.get("spill_result_above_bytes", RLM_DEFAULT_SPILL_RESULT_ABOVE_BYTES)
            ),
            max_registry_bytes=int(data.get("max_registry_bytes", RLM_DEFAULT_MAX_REGISTRY_BYTES)),
            max_spill_bytes=int(data.get("max_spill_bytes", RLM_DEFAULT_MAX_SPILL_BYTES)),
        )

    def registry_kwargs(self) -> dict[str, int]:
        """The result-registry fields, keyed as ``RunConfigDTO`` names them."""
        return {
            "result_preview_chars": self.result_preview_chars,
            "spill_result_above_bytes": self.spill_result_above_bytes,
            "max_registry_bytes": self.max_registry_bytes,
            "max_spill_bytes": self.max_spill_bytes,
        }


@dataclass
class MonitoringConfig:
    """Monitoring and telemetry configuration."""

    enable_telemetry: bool = False
    log_level: str = "INFO"
    export_format: str = "json"

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "enable_telemetry": self.enable_telemetry,
            "log_level": self.log_level,
            "export_format": self.export_format,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MonitoringConfig":
        """Create config from dictionary."""
        return cls(
            enable_telemetry=data.get("enable_telemetry", False),
            log_level=data.get("log_level", "INFO"),
            export_format=data.get("export_format", "json"),
        )


class RLMConfig:
    """Main RLMKit configuration."""

    # Default config file locations (in priority order)
    CONFIG_SEARCH_PATHS = [
        f"./{CONFIG_FILE_STEM}.yaml",
        f"./{CONFIG_FILE_STEM}.json",
        *[f"./{stem}.{ext}" for stem in LEGACY_CONFIG_FILE_STEMS for ext in ("yaml", "json")],
        f"~/{STATE_DIR_NAME}/config.yaml",
        f"~/{STATE_DIR_NAME}/config.json",
        f"/etc/{DIST_NAME}/config.yaml",
        f"/etc/{DIST_NAME}/config.json",
    ]

    def __init__(
        self,
        security: SecurityConfig | None = None,
        execution: ExecutionConfig | None = None,
        monitoring: MonitoringConfig | None = None,
        llm: LLMConfig | None = None,
    ):
        """
        Initialize RLMKit configuration.

        Args:
            security: Security configuration
            execution: Execution configuration
            monitoring: Monitoring configuration
            llm: LLM provider configuration
        """
        self.security = security or SecurityConfig()
        self.execution = execution or ExecutionConfig()
        self.monitoring = monitoring or MonitoringConfig()
        self.llm = llm or LLMConfig()

    @classmethod
    def load(cls, config_path: str | None = None) -> "RLMConfig":
        """
        Load configuration from file.

        Search order:
        1. Explicit config_path parameter
        2. RLM_STUDIO_CONFIG_PATH environment variable (legacy RLMKIT_CONFIG_PATH honoured)
        3. Default search paths (project, user home, system)

        Args:
            config_path: Optional explicit path to config file

        Returns:
            RLMConfig instance
        """
        # Try explicit path first
        if config_path:
            return cls._load_from_file(config_path)

        # Try environment variable
        env_path = env("CONFIG_PATH")
        if env_path and Path(env_path).exists():
            return cls._load_from_file(env_path)

        # Try default search paths
        for path_str in cls.CONFIG_SEARCH_PATHS:
            path = Path(path_str).expanduser()
            if path.exists():
                return cls._load_from_file(str(path))

        # No config file found, use defaults
        return cls()

    @classmethod
    def _load_from_file(cls, filepath: str) -> "RLMConfig":
        """Load configuration from a specific file."""
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        # Read file content
        with open(path) as f:
            content = f.read()

        # Parse based on extension
        if path.suffix in [".yaml", ".yml"]:
            data = yaml.safe_load(content)
        elif path.suffix == ".json":
            data = json.loads(content)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

        # Create config from data
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RLMConfig":
        """Create config from dictionary."""
        security = SecurityConfig.from_dict(data.get("security", {}))
        execution = ExecutionConfig.from_dict(data.get("execution", {}))
        monitoring = MonitoringConfig.from_dict(data.get("monitoring", {}))
        llm = LLMConfig.from_dict(data.get("llm", {}))

        return cls(security=security, execution=execution, monitoring=monitoring, llm=llm)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "security": self.security.to_dict(),
            "execution": self.execution.to_dict(),
            "monitoring": self.monitoring.to_dict(),
            "llm": self.llm.to_dict(),
        }

    def save(self, filepath: str) -> None:
        """
        Save configuration to file.

        Args:
            filepath: Path to save config file
        """
        path = Path(filepath)
        data = self.to_dict()

        # Save based on extension
        if path.suffix in [".yaml", ".yml"]:
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        elif path.suffix == ".json":
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")


# Global default configuration instance
_default_config: RLMConfig | None = None


def get_default_config() -> RLMConfig:
    """
    Get the default configuration instance.

    Lazily loads configuration on first access.

    Returns:
        RLMConfig instance
    """
    global _default_config
    if _default_config is None:
        _default_config = RLMConfig.load()
    return _default_config


def set_default_config(config: RLMConfig) -> None:
    """
    Set the default configuration instance.

    Args:
        config: RLMConfig to use as default
    """
    global _default_config
    _default_config = config
