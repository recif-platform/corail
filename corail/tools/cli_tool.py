"""CLI Tool Executor — runs local CLI commands securely."""

import asyncio
import re

from corail.tools.base import ToolDefinition, ToolExecutor, ToolParameter, ToolResult

# Characters that could enable shell injection
_DANGEROUS_PATTERN = re.compile(r"[;|`$(){}]")


def _sanitize_value(value: str) -> str:
    """Strip characters that could be used for shell injection."""
    return _DANGEROUS_PATTERN.sub("", value)


class CLIToolExecutor(ToolExecutor):
    """Executes local CLI commands via subprocess. Never uses shell=True."""

    def __init__(
        self,
        name: str,
        description: str,
        binary: str,
        parameters: list[ToolParameter] | None = None,
        allowed_commands: list[str] | None = None,
        env: dict[str, str] | None = None,
        timeout: float = 30.0,
        json_flag: str = "--json",
    ) -> None:
        self._name = name
        self._description = description
        self._binary = binary
        self._parameters = parameters or []
        self._allowed_commands = allowed_commands
        self._env = env
        self._timeout = timeout
        self._json_flag = json_flag

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description=self._description,
            parameters=self._parameters,
        )

    async def execute(self, **kwargs: object) -> ToolResult:
        try:
            args = self._build_args(kwargs)
        except ValueError as e:
            return ToolResult(success=False, output="", error=str(e))

        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._env,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=self._timeout,
            )
        except TimeoutError:
            process.kill()
            await process.communicate()
            return ToolResult(success=False, output="", error="Command timed out")
        except FileNotFoundError:
            return ToolResult(success=False, output="", error=f"Binary not found: {self._binary}")

        stdout = stdout_bytes.decode(errors="replace")
        stderr = stderr_bytes.decode(errors="replace")

        if process.returncode == 0:
            return ToolResult(success=True, output=stdout[:4000])
        return ToolResult(success=False, output=stdout[:2000], error=stderr[:2000])

    def _build_args(self, kwargs: dict[str, object]) -> list[str]:
        """Build the subprocess argument list from kwargs."""
        args: list[str] = [self._binary]

        # Handle subcommand
        command = kwargs.pop("command", None)
        if command is not None:
            command = str(command)
            if self._allowed_commands is not None and command not in self._allowed_commands:
                allowed = ", ".join(self._allowed_commands)
                msg = f"Command '{command}' not allowed. Allowed: {allowed}"
                raise ValueError(msg)
            args.append(_sanitize_value(command))

        # Map remaining kwargs to CLI flags
        for key, value in kwargs.items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    args.append(flag)
            else:
                sanitized = _sanitize_value(str(value))
                args.extend([flag, sanitized])

        return args
