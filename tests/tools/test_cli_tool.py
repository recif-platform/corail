"""Tests for CLIToolExecutor."""

import os
import sys
import tempfile

import pytest

from corail.tools.base import ToolParameter
from corail.tools.cli_tool import CLIToolExecutor

# Helper: path to a tiny script that echoes its arguments as JSON
_SCRIPT = os.path.join(os.path.dirname(__file__), "_helper.py")


@pytest.fixture(autouse=True, scope="module")
def _create_helper_script():
    """Create a helper Python script used by multiple tests."""
    content = (
        "import json, sys\n"
        "print(json.dumps({'args': sys.argv[1:], 'status': 'ok'}))\n"
    )
    os.makedirs(os.path.dirname(_SCRIPT), exist_ok=True)
    with open(_SCRIPT, "w") as f:
        f.write(content)
    yield
    if os.path.exists(_SCRIPT):
        os.remove(_SCRIPT)


class TestCLIToolDefinition:
    def test_definition_basic(self):
        tool = CLIToolExecutor(name="git", description="Run git commands", binary="git")
        d = tool.definition()
        assert d.name == "git"
        assert d.description == "Run git commands"
        assert d.parameters == []

    def test_definition_with_parameters(self):
        params = [
            ToolParameter(name="command", type="string", description="Git subcommand"),
            ToolParameter(name="branch", type="string", description="Branch name", required=False),
        ]
        tool = CLIToolExecutor(name="git", description="Run git", binary="git", parameters=params)
        d = tool.definition()
        assert len(d.parameters) == 2
        assert d.parameters[0].name == "command"
        assert d.parameters[1].required is False

    def test_name_property(self):
        tool = CLIToolExecutor(name="kubectl", description="K8s CLI", binary="kubectl")
        assert tool.name == "kubectl"


class TestCLIToolExecution:
    async def test_basic_echo(self):
        tool = CLIToolExecutor(name="echo", description="Echo text", binary="echo")
        result = await tool.execute(message="hello")
        assert result.success
        assert "hello" in result.output

    async def test_subcommand_execution(self):
        tool = CLIToolExecutor(name="echo", description="Echo", binary="echo")
        result = await tool.execute(command="status", verbose=True)
        assert result.success
        assert "status" in result.output

    async def test_json_output(self):
        """Run the helper script and verify JSON output."""
        tool = CLIToolExecutor(
            name="helper",
            description="Helper script",
            binary=sys.executable,
        )
        result = await tool.execute(command=_SCRIPT, name="test")
        assert result.success
        assert '"status"' in result.output
        assert '"ok"' in result.output

    async def test_exit_code_nonzero(self):
        """A script that writes to stdout/stderr then exits non-zero."""
        script = (
            "import sys\n"
            "print('out-marker')\n"
            "print('err-marker', file=sys.stderr)\n"
            "sys.exit(1)\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            script_path = f.name
        try:
            tool = CLIToolExecutor(name="fail", description="Fail", binary=sys.executable)
            result = await tool.execute(command=script_path)
            assert not result.success
            assert "out-marker" in result.output
            assert "err-marker" in result.error
        finally:
            os.unlink(script_path)

    async def test_binary_not_found(self):
        tool = CLIToolExecutor(
            name="missing",
            description="Missing binary",
            binary="/nonexistent/binary",
        )
        result = await tool.execute()
        assert not result.success
        assert "not found" in result.error.lower()

    async def test_output_truncation(self):
        """Output beyond 4000 chars is truncated."""
        script = "print('x' * 10000)\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            script_path = f.name
        try:
            tool = CLIToolExecutor(
                name="big_output",
                description="Lots of output",
                binary=sys.executable,
            )
            result = await tool.execute(command=script_path)
            assert result.success
            assert len(result.output) <= 4000
        finally:
            os.unlink(script_path)


class TestCLIToolAllowedCommands:
    async def test_allowed_command_passes(self):
        tool = CLIToolExecutor(
            name="echo",
            description="Echo",
            binary="echo",
            allowed_commands=["hello", "world"],
        )
        result = await tool.execute(command="hello")
        assert result.success
        assert "hello" in result.output

    async def test_rejected_command(self):
        tool = CLIToolExecutor(
            name="echo",
            description="Echo",
            binary="echo",
            allowed_commands=["safe"],
        )
        result = await tool.execute(command="dangerous")
        assert not result.success
        assert "not allowed" in result.error.lower()

    async def test_no_whitelist_allows_all(self):
        tool = CLIToolExecutor(
            name="echo",
            description="Echo",
            binary="echo",
        )
        result = await tool.execute(command="anything")
        assert result.success


class TestCLIToolTimeout:
    async def test_timeout(self):
        script = "import time; time.sleep(10)\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            script_path = f.name
        try:
            tool = CLIToolExecutor(
                name="slow",
                description="Slow command",
                binary=sys.executable,
                timeout=0.5,
            )
            result = await tool.execute(command=script_path)
            assert not result.success
            assert "timed out" in result.error.lower()
        finally:
            os.unlink(script_path)


class TestCLIToolSecurity:
    async def test_semicolon_stripped(self):
        tool = CLIToolExecutor(name="echo", description="Echo", binary="echo")
        result = await tool.execute(message="hello; rm -rf /")
        assert result.success
        assert ";" not in result.output

    async def test_backtick_stripped(self):
        tool = CLIToolExecutor(name="echo", description="Echo", binary="echo")
        result = await tool.execute(message="`whoami`")
        assert result.success
        assert "`" not in result.output

    async def test_pipe_stripped(self):
        tool = CLIToolExecutor(name="echo", description="Echo", binary="echo")
        result = await tool.execute(message="hello | cat /etc/passwd")
        assert result.success
        assert "|" not in result.output

    async def test_dollar_stripped(self):
        tool = CLIToolExecutor(name="echo", description="Echo", binary="echo")
        result = await tool.execute(message="$(whoami)")
        assert result.success
        assert "$" not in result.output
        assert "(" not in result.output

    async def test_boolean_true_becomes_flag(self):
        tool = CLIToolExecutor(name="echo", description="Echo", binary="echo")
        result = await tool.execute(verbose=True)
        assert result.success
        assert "--verbose" in result.output

    async def test_boolean_false_omitted(self):
        tool = CLIToolExecutor(name="echo", description="Echo", binary="echo")
        result = await tool.execute(verbose=False, message="hi")
        assert result.success
        assert "--verbose" not in result.output
        assert "hi" in result.output

    async def test_underscore_to_hyphen(self):
        tool = CLIToolExecutor(name="echo", description="Echo", binary="echo")
        result = await tool.execute(dry_run=True)
        assert result.success
        assert "--dry-run" in result.output


class TestCLIToolFactory:
    def test_factory_creates_cli_tool(self):
        from corail.tools.factory import ToolFactory

        tool = ToolFactory.create("cli", name="test", description="A test CLI tool", binary="echo")
        assert tool.name == "test"
        assert isinstance(tool, CLIToolExecutor)

    def test_factory_available_includes_cli(self):
        from corail.tools.factory import ToolFactory

        assert "cli" in ToolFactory.available()

    def test_factory_unknown_raises(self):
        from corail.tools.factory import ToolFactory

        with pytest.raises(ValueError, match="Unknown tool type"):
            ToolFactory.create("nonexistent")
