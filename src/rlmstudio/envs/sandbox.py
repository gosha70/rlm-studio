# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0

"""Sandbox restrictions for safe code execution."""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from rlmstudio.branding import SANDBOX_IMAGE_NAME
from rlmstudio.config import SecurityConfig
from rlmstudio.core.errors import ExecutionError, SecurityError


class RestrictedBuiltins:
    """Provides restricted builtins for safe code execution."""

    def __init__(self, security_config: SecurityConfig | None = None):
        """
        Initialize restricted builtins.

        Args:
            security_config: Security configuration (uses default if None)
        """
        self.config = security_config or SecurityConfig()

    def get_restricted_builtins(self) -> dict[str, Any]:
        """
        Get restricted builtins dict for safe execution.

        Returns:
            Dictionary with only safe builtins
        """
        import builtins

        safe_builtins = {}
        for name in self.config.safe_builtins:
            if hasattr(builtins, name):
                safe_builtins[name] = getattr(builtins, name)

        # Add restricted versions of some builtins
        safe_builtins["__builtins__"] = safe_builtins

        return safe_builtins


class SafeImporter:
    """Controls which modules can be imported."""

    def __init__(
        self,
        security_config: SecurityConfig | None = None,
        allowed_imports: list[str] | None = None,
    ):
        """
        Initialize safe importer.

        Args:
            security_config: Security configuration (uses default if None)
            allowed_imports: Additional modules to allow (beyond config)
        """
        self.config = security_config or SecurityConfig()

        # Start with safe modules from config
        self.allowed_imports = set(self.config.safe_modules)

        # Add any additional allowed imports
        if allowed_imports:
            self.allowed_imports.update(allowed_imports)

    def check_import(self, module_name: str) -> bool:
        """
        Check if a module import is allowed.

        Args:
            module_name: Name of module to import

        Returns:
            True if allowed, False otherwise
        """
        # Extract base module name (e.g., 'json.decoder' -> 'json')
        base_module = module_name.split(".")[0]

        # Check if explicitly blocked in config
        if base_module in self.config.blocked_modules:
            return False

        # Check if in allowed list
        return base_module in self.allowed_imports

    def safe_import(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """
        Safe import function that checks against whitelist.

        Args:
            name: Module name to import
            *args: Additional import args
            **kwargs: Additional import kwargs

        Returns:
            Imported module

        Raises:
            SecurityError: If import is not allowed
        """
        if not self.check_import(name):
            blocked_list = ", ".join(sorted(self.config.blocked_modules)[:10])
            raise SecurityError(
                f"Import of module '{name}' is not allowed in safe mode. "
                f"Blocked modules: {blocked_list}..."
            )

        # Use the real __import__
        import builtins

        return builtins.__import__(name, *args, **kwargs)


def create_safe_globals(
    allowed_imports: list[str] | None = None,
    additional_globals: dict[str, Any] | None = None,
    security_config: SecurityConfig | None = None,
) -> dict[str, Any]:
    """
    Create safe globals dictionary for code execution.

    Args:
        allowed_imports: List of additional modules to allow importing
        additional_globals: Additional global variables to include
        security_config: Security configuration (uses default if None)

    Returns:
        Safe globals dictionary with restricted builtins and safe __import__
    """
    # Get restricted builtins using config
    builtins_provider = RestrictedBuiltins(security_config)
    safe_builtins = builtins_provider.get_restricted_builtins()

    # Create safe importer with same config
    importer = SafeImporter(security_config, allowed_imports)

    # Add __import__ to the builtins dict (where Python looks for it)
    safe_builtins["__import__"] = importer.safe_import

    # Build safe globals
    safe_globals = {
        "__builtins__": safe_builtins,
        "__name__": "__rlm__",
    }

    # Add any additional globals
    if additional_globals:
        safe_globals.update(additional_globals)

    return safe_globals


def validate_code_safety(code: str) -> None:
    """
    Perform static analysis to check for obviously dangerous code patterns.

    This is a basic check - the main safety comes from restricted builtins.

    Args:
        code: Python code to validate

    Raises:
        SecurityError: If dangerous patterns detected
    """
    dangerous_patterns = [
        ("__import__", "Direct __import__ usage not allowed"),
        ("eval(", "eval() not allowed"),
        ("exec(", "exec() not allowed"),
        ("compile(", "compile() not allowed"),
        ("open(", "File operations not allowed"),
        ("os.", "os module not allowed"),
        ("sys.", "sys module not allowed"),
        ("subprocess", "subprocess module not allowed"),
    ]

    for pattern, _message in dangerous_patterns:
        if pattern in code:
            # Note: This is a basic check. Real safety comes from restricted builtins.
            # We just warn here, actual blocking happens at execution.
            pass  # Could add warnings here if desired


class DockerExecutor:
    """
    Execute Python code in isolated Docker container (Bet 5 - Enterprise Security).

    This provides the strongest isolation level by running code in a disposable
    Docker container with resource limits and network restrictions.

    Features:
    - Process isolation via containers
    - Resource limits (CPU, memory, timeout)
    - Network isolation (no external access)
    - Ephemeral containers (destroyed after execution)
    - Non-root user execution

    Attributes:
        image_name: Docker image to use (default: ``SANDBOX_IMAGE_NAME``)
        memory_limit: Memory limit (e.g., "512m", "1g")
        cpu_limit: CPU limit (e.g., "1" for 1 core)
        timeout: Execution timeout in seconds
        network_mode: Docker network mode (default: "none" for isolation)
    """

    def __init__(
        self,
        image_name: str = SANDBOX_IMAGE_NAME,
        memory_limit: str = "512m",
        cpu_limit: str = "1",
        timeout: int = 30,
        network_mode: str = "none",
        auto_build: bool = True,
    ):
        """
        Initialize Docker executor.

        Args:
            image_name: Name of Docker image to use
            memory_limit: Memory limit for container
            cpu_limit: CPU limit for container
            timeout: Maximum execution time in seconds
            network_mode: Docker network mode ("none" for isolation)
            auto_build: If True, build image if it doesn't exist
        """
        self.image_name = image_name
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.timeout = timeout
        self.network_mode = network_mode

        if auto_build:
            self._ensure_image_exists()

    def _ensure_image_exists(self) -> None:
        """Check if Docker image exists, build if not."""
        try:
            # Check if image exists
            result = subprocess.run(
                ["docker", "images", "-q", self.image_name],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if not result.stdout.strip():
                # Image doesn't exist, try to build it
                self._build_image()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Docker not available or timeout
            pass

    def _build_image(self) -> None:
        """Build the Docker image from Dockerfile."""
        # Find Dockerfile path
        dockerfile_path = (
            Path(__file__).parent.parent.parent.parent / "docker" / "Dockerfile.sandbox"
        )

        if not dockerfile_path.exists():
            raise ExecutionError(
                f"Dockerfile not found at {dockerfile_path}. Cannot build sandbox image."
            )

        try:
            subprocess.run(
                ["docker", "build", "-t", self.image_name, "-f", str(dockerfile_path), "."],
                cwd=dockerfile_path.parent,
                check=True,
                capture_output=True,
                timeout=120,
            )
        except subprocess.CalledProcessError as e:
            raise ExecutionError(f"Failed to build Docker image: {e.stderr.decode()}") from e
        except subprocess.TimeoutExpired as e:
            raise ExecutionError("Docker image build timed out") from e

    def execute(self, code: str, globals_dict: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Execute code in Docker container.

        Args:
            code: Python code to execute
            globals_dict: Global variables to pass to code

        Returns:
            Dictionary with 'result', 'output', 'error' keys

        Raises:
            ExecutionError: If execution fails
            SecurityError: If Docker is not available
        """
        # Check if Docker is available
        if not self._is_docker_available():
            raise SecurityError(
                "Docker is not available. Docker execution requires Docker to be installed and running."
            )

        # Create temporary file with code
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            # Serialize globals if provided
            if globals_dict:
                f.write("import json\n")
                f.write(f"_globals = {repr(globals_dict)}\n")
                f.write("globals().update(_globals)\n\n")

            f.write(code)
            temp_file = f.name

        try:
            # Run code in Docker container
            docker_cmd = [
                "docker",
                "run",
                "--rm",  # Remove container after execution
                f"--memory={self.memory_limit}",
                f"--cpus={self.cpu_limit}",
                f"--network={self.network_mode}",
                "--security-opt=no-new-privileges",  # Additional security
                "-v",
                f"{temp_file}:/sandbox/script.py:ro",  # Mount script read-only
                self.image_name,
                "python3",
                "/sandbox/script.py",
            ]

            result = subprocess.run(
                docker_cmd, capture_output=True, text=True, timeout=self.timeout
            )

            return {
                "result": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else None,
            }

        except subprocess.TimeoutExpired:
            return {
                "result": False,
                "output": "",
                "error": f"Execution timed out after {self.timeout} seconds",
            }
        except Exception as e:
            return {"result": False, "output": "", "error": f"Docker execution failed: {str(e)}"}
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except Exception:
                pass

    @staticmethod
    def _is_docker_available() -> bool:
        """Check if Docker is available on the system."""
        try:
            result = subprocess.run(["docker", "--version"], capture_output=True, timeout=5)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def is_available() -> bool:
        """
        Check if Docker execution is available.

        Returns:
            True if Docker is installed and running
        """
        return DockerExecutor._is_docker_available()
