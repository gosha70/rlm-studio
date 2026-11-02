"""Tests for configuration system."""

import os
import tempfile
from pathlib import Path

from rlmstudio.config import (
    ExecutionConfig,
    MonitoringConfig,
    RLMConfig,
    SecurityConfig,
    get_default_config,
    set_default_config,
)


class TestSecurityConfig:
    """Test SecurityConfig functionality."""

    def test_default_config(self):
        """Test default security configuration."""
        config = SecurityConfig()

        # Check default blocked modules
        assert "os" in config.blocked_modules
        assert "sys" in config.blocked_modules
        assert "subprocess" in config.blocked_modules

        # Check default safe modules
        assert "json" in config.safe_modules
        assert "re" in config.safe_modules
        assert "math" in config.safe_modules

    def test_add_safe_module(self):
        """Test adding a safe module."""
        config = SecurityConfig()
        config.add_safe_module("pandas")

        assert "pandas" in config.safe_modules
        assert "pandas" not in config.blocked_modules

    def test_add_safe_module_removes_from_blocked(self):
        """Test that adding safe module removes it from blocked."""
        config = SecurityConfig()
        config.add_blocked_module("test_module")
        assert "test_module" in config.blocked_modules

        config.add_safe_module("test_module")
        assert "test_module" in config.safe_modules
        assert "test_module" not in config.blocked_modules

    def test_add_blocked_module(self):
        """Test adding a blocked module."""
        config = SecurityConfig()
        config.add_blocked_module("dangerous_module")

        assert "dangerous_module" in config.blocked_modules
        assert "dangerous_module" not in config.safe_modules

    def test_is_module_allowed(self):
        """Test module allowance checking."""
        config = SecurityConfig()

        # Safe module should be allowed
        assert config.is_module_allowed("json")
        assert config.is_module_allowed("math")

        # Blocked module should not be allowed
        assert not config.is_module_allowed("os")
        assert not config.is_module_allowed("sys")

        # Unknown module should not be allowed
        assert not config.is_module_allowed("unknown_module")

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = SecurityConfig()
        data = config.to_dict()

        assert "blocked_modules" in data
        assert "safe_modules" in data
        assert "blocked_builtins" in data
        assert "safe_builtins" in data

        # Check that lists are sorted
        assert isinstance(data["blocked_modules"], list)
        assert isinstance(data["safe_modules"], list)

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "blocked_modules": ["os", "sys"],
            "safe_modules": ["json", "re"],
            "blocked_builtins": ["open", "eval"],
            "safe_builtins": ["print", "len"],
        }

        config = SecurityConfig.from_dict(data)

        assert "os" in config.blocked_modules
        assert "json" in config.safe_modules
        assert "open" in config.blocked_builtins
        assert "print" in config.safe_builtins


class TestExecutionConfig:
    """Test ExecutionConfig functionality."""

    def test_default_config(self):
        """Test default execution configuration."""
        config = ExecutionConfig()

        assert config.default_timeout == 5.0
        assert config.max_output_chars == 10000
        assert config.default_safe_mode is True

    def test_custom_config(self):
        """Test custom execution configuration."""
        config = ExecutionConfig(
            default_timeout=10.0, max_output_chars=50000, default_safe_mode=True
        )

        assert config.default_timeout == 10.0
        assert config.max_output_chars == 50000
        assert config.default_safe_mode is True

    def test_to_from_dict(self):
        """Test dict conversion roundtrip."""
        config1 = ExecutionConfig(default_timeout=3.0, max_output_chars=5000)
        data = config1.to_dict()
        config2 = ExecutionConfig.from_dict(data)

        assert config2.default_timeout == 3.0
        assert config2.max_output_chars == 5000


class TestMonitoringConfig:
    """Test MonitoringConfig functionality."""

    def test_default_config(self):
        """Test default monitoring configuration."""
        config = MonitoringConfig()

        assert config.enable_telemetry is False
        assert config.log_level == "INFO"
        assert config.export_format == "json"

    def test_custom_config(self):
        """Test custom monitoring configuration."""
        config = MonitoringConfig(enable_telemetry=True, log_level="DEBUG", export_format="csv")

        assert config.enable_telemetry is True
        assert config.log_level == "DEBUG"
        assert config.export_format == "csv"


class TestRLMConfig:
    """Test main RLMConfig functionality."""

    def test_default_config(self):
        """Test default configuration."""
        config = RLMConfig()

        assert isinstance(config.security, SecurityConfig)
        assert isinstance(config.execution, ExecutionConfig)
        assert isinstance(config.monitoring, MonitoringConfig)

    def test_custom_configs(self):
        """Test with custom sub-configs."""
        security = SecurityConfig()
        security.add_safe_module("pandas")

        execution = ExecutionConfig(default_timeout=10.0)
        monitoring = MonitoringConfig(enable_telemetry=True)

        config = RLMConfig(security=security, execution=execution, monitoring=monitoring)

        assert "pandas" in config.security.safe_modules
        assert config.execution.default_timeout == 10.0
        assert config.monitoring.enable_telemetry is True

    def test_old_config_with_pricing_block_still_loads(self):
        """Removing the pricing schema must not make an older file fatal.

        Configs written before the block was deleted still carry it; loading
        ignores the unknown key and a re-save drops it.
        """
        data = {
            "llm": {
                "default_provider": {"provider": "openai", "model": "gpt-4o"},
                "pricing": {"gpt-4": {"input_cost_per_1m": 30.0, "output_cost_per_1m": 60.0}},
            }
        }

        config = RLMConfig.from_dict(data)

        assert config.llm.default_provider is not None
        assert config.llm.default_provider.model == "gpt-4o"
        assert "pricing" not in config.llm.to_dict()

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = RLMConfig()
        data = config.to_dict()

        assert "security" in data
        assert "execution" in data
        assert "monitoring" in data

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "security": {
                "blocked_modules": ["os"],
                "safe_modules": ["json"],
                "blocked_builtins": ["open"],
                "safe_builtins": ["print"],
            },
            "execution": {
                "default_timeout": 3.0,
                "max_output_chars": 5000,
                "default_safe_mode": True,
            },
            "monitoring": {
                "enable_telemetry": True,
                "log_level": "DEBUG",
                "export_format": "csv",
            },
        }

        config = RLMConfig.from_dict(data)

        assert "os" in config.security.blocked_modules
        assert config.execution.default_timeout == 3.0
        assert config.monitoring.enable_telemetry is True

    def test_save_and_load_yaml(self):
        """Test saving and loading YAML config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"

            # Create and save config
            config1 = RLMConfig()
            config1.execution.default_timeout = 7.5
            config1.save(str(config_path))

            # Load config
            config2 = RLMConfig.load(str(config_path))

            assert config2.execution.default_timeout == 7.5

    def test_save_and_load_json(self):
        """Test saving and loading JSON config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.json"

            # Create and save config
            config1 = RLMConfig()
            config1.monitoring.enable_telemetry = True
            config1.save(str(config_path))

            # Load config
            config2 = RLMConfig.load(str(config_path))

            assert config2.monitoring.enable_telemetry is True

    def test_load_nonexistent_file(self):
        """Test loading non-existent file returns defaults."""
        config = RLMConfig.load()  # No file exists

        # Should return default config
        assert config.execution.default_timeout == 5.0

    def test_load_with_env_variable(self):
        """Test loading from environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "env_config.yaml"

            # Create config file
            config1 = RLMConfig()
            config1.execution.max_output_chars = 99999
            config1.save(str(config_path))

            # Set environment variable
            os.environ["RLM_STUDIO_CONFIG_PATH"] = str(config_path)

            try:
                # Load should use env variable
                config2 = RLMConfig.load()
                assert config2.execution.max_output_chars == 99999
            finally:
                # Clean up
                del os.environ["RLM_STUDIO_CONFIG_PATH"]


class TestConfigIntegration:
    """Test integration with sandbox."""

    def test_security_config_with_sandbox(self):
        """Test that SecurityConfig works with sandbox."""
        from rlmstudio.envs.sandbox import RestrictedBuiltins, SafeImporter

        # Create custom config
        config = SecurityConfig()
        config.add_safe_module("custom_module")

        # Use with RestrictedBuiltins
        builtins_provider = RestrictedBuiltins(config)
        safe_builtins = builtins_provider.get_restricted_builtins()

        assert "print" in safe_builtins
        assert "open" not in safe_builtins

        # Use with SafeImporter
        importer = SafeImporter(config, allowed_imports=["another_module"])

        assert importer.check_import("json")  # From config
        assert importer.check_import("custom_module")  # Added to config
        assert importer.check_import("another_module")  # From allowed_imports
        assert not importer.check_import("os")  # Blocked


class TestGlobalConfig:
    """Test global config management."""

    def test_get_default_config(self):
        """Test getting default config."""
        config = get_default_config()
        assert isinstance(config, RLMConfig)

    def test_set_default_config(self):
        """Test setting default config."""
        # Create custom config
        custom_config = RLMConfig()
        custom_config.execution.default_timeout = 99.0

        # Set as default
        set_default_config(custom_config)

        # Get default should return custom
        config = get_default_config()
        assert config.execution.default_timeout == 99.0

        # Reset to default for other tests
        set_default_config(RLMConfig())
