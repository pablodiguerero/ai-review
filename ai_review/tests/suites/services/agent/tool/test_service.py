import re
import subprocess
from pathlib import Path

import pytest

from ai_review.config import settings
from ai_review.services.agent.tool.service import AgentToolService, TRUNCATION_HINT
from ai_review.services.policy.service import PolicyService
from ai_review.tests.fixtures.services.policy import FakePolicyService


@pytest.mark.asyncio
async def test_execute_runs_allowed_command(
        tmp_path: Path,
        agent_tool_service: AgentToolService,
        fake_policy_service: FakePolicyService,
) -> None:
    (tmp_path / "sample.txt").write_text("hello", encoding="utf-8")
    fake_policy_service.responses["should_agent_run_command"] = True

    result = await agent_tool_service.execute("cat sample.txt")

    assert result.executed is True
    assert result.command == "cat sample.txt"
    assert "exit_code: 0" in result.output
    assert "hello" in result.output
    assert any(call[0] == "should_agent_run_command" for call in fake_policy_service.calls)


@pytest.mark.asyncio
async def test_execute_blocks_disallowed_command(
        agent_tool_service: AgentToolService,
        fake_policy_service: FakePolicyService,
) -> None:
    fake_policy_service.responses["should_agent_run_command"] = False

    result = await agent_tool_service.execute("cat sample.txt")

    assert result.executed is False
    assert "blocked by policy" in result.output.lower()
    assert "allowed:" in result.output.lower()


@pytest.mark.asyncio
async def test_execute_blocks_shell_operators_with_actionable_message(
        agent_tool_service: AgentToolService,
        fake_policy_service: FakePolicyService,
) -> None:
    fake_policy_service.responses["should_agent_run_command"] = False
    fake_policy_service.responses["command_has_shell_operators"] = True

    result = await agent_tool_service.execute("ls && cat foo")

    assert result.executed is False
    assert "shell operators are not supported" in result.output
    assert "no pipes, no &&, no redirects" in result.output
    assert "ls && cat foo" in result.output


@pytest.mark.asyncio
async def test_execute_blocked_message_reflects_custom_allowlist(
        monkeypatch: pytest.MonkeyPatch,
        agent_tool_service: AgentToolService,
        fake_policy_service: FakePolicyService,
) -> None:
    fake_policy_service.responses["should_agent_run_command"] = False
    custom_patterns = [re.compile(r"^only-this(?:\s+.*)?$"), re.compile(r"^and-this(?:\s+.*)?$")]
    monkeypatch.setattr(settings.agent, "allow_commands", custom_patterns)

    result = await agent_tool_service.execute("cat sample.txt")

    assert result.executed is False
    assert "only-this(?:\\s+.*)?$" in result.output
    assert "and-this(?:\\s+.*)?$" in result.output


@pytest.mark.asyncio
async def test_execute_blocks_newline_bypass_with_generic_message(tmp_path: Path) -> None:
    tool_service = AgentToolService(policy=PolicyService(), repo_dir=tmp_path)

    result = await tool_service.execute("find\n. -exec rm {} +")

    assert result.executed is False
    assert "shell operators are not supported" not in result.output
    assert "blocked by policy" in result.output.lower()
    assert "allowed:" in result.output.lower()


@pytest.mark.asyncio
async def test_execute_runs_in_repo_directory(
        tmp_path: Path,
        agent_tool_service: AgentToolService,
        fake_policy_service: FakePolicyService,
) -> None:
    (tmp_path / "visible.txt").write_text("ok", encoding="utf-8")
    fake_policy_service.responses["should_agent_run_command"] = True

    result = await agent_tool_service.execute("ls")

    assert "visible.txt" in result.output


@pytest.mark.asyncio
async def test_execute_truncates_large_output_and_appends_hint(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        agent_tool_service: AgentToolService,
        fake_policy_service: FakePolicyService,
) -> None:
    huge = "x" * 5_000
    (tmp_path / "big.txt").write_text(huge, encoding="utf-8")
    fake_policy_service.responses["should_agent_run_command"] = True
    monkeypatch.setattr(settings.agent, "max_command_output_chars", 1_000)
    agent_tool_service.max_command_output_chars = 1_000

    result = await agent_tool_service.execute("cat big.txt")

    assert result.executed is True
    assert "output truncated" in result.output.lower()
    assert "Output truncated: read the file in ranges" in result.output


@pytest.mark.asyncio
async def test_execute_truncation_hint_fits_within_output_limit(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        agent_tool_service: AgentToolService,
        fake_policy_service: FakePolicyService,
) -> None:
    huge = "x" * 5_000
    (tmp_path / "big.txt").write_text(huge, encoding="utf-8")
    fake_policy_service.responses["should_agent_run_command"] = True
    limit = 200
    monkeypatch.setattr(settings.agent, "max_command_output_chars", limit)
    agent_tool_service.max_command_output_chars = limit

    result = await agent_tool_service.execute("cat big.txt")

    assert result.output.endswith(TRUNCATION_HINT)
    content_before_hint = result.output[:-len(TRUNCATION_HINT)]
    assert len(content_before_hint) < limit
    assert len(result.output) <= limit


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [200, 1_000, 8_000])
async def test_execute_truncated_output_never_exceeds_limit(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        agent_tool_service: AgentToolService,
        fake_policy_service: FakePolicyService,
        limit: int,
) -> None:
    huge = "x" * 20_000
    (tmp_path / "big.txt").write_text(huge, encoding="utf-8")
    fake_policy_service.responses["should_agent_run_command"] = True
    monkeypatch.setattr(settings.agent, "max_command_output_chars", limit)
    agent_tool_service.max_command_output_chars = limit

    result = await agent_tool_service.execute("cat big.txt")

    assert len(result.output) <= limit


@pytest.mark.asyncio
async def test_execute_rejects_empty_command(agent_tool_service: AgentToolService) -> None:
    result = await agent_tool_service.execute("   ")
    assert result.executed is False
    assert "empty command" in result.output.lower()


@pytest.mark.asyncio
async def test_execute_rejects_none_command(agent_tool_service: AgentToolService) -> None:
    result = await agent_tool_service.execute(None)
    assert result.executed is False
    assert "empty command" in result.output.lower()


@pytest.mark.asyncio
async def test_execute_returns_parse_error_for_invalid_shell_syntax(
        agent_tool_service: AgentToolService,
        fake_policy_service: FakePolicyService,
) -> None:
    fake_policy_service.responses["should_agent_run_command"] = True

    result = await agent_tool_service.execute('"unterminated')

    assert result.executed is False
    assert "parse error" in result.output.lower()


@pytest.mark.asyncio
async def test_execute_returns_timeout_error(
        monkeypatch: pytest.MonkeyPatch,
        agent_tool_service: AgentToolService,
        fake_policy_service: FakePolicyService,
) -> None:
    fake_policy_service.responses["should_agent_run_command"] = True

    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["ls"], timeout=0.01)

    monkeypatch.setattr(subprocess, "run", _raise_timeout)
    result = await agent_tool_service.execute("ls")

    assert result.executed is False
    assert "timeout" in result.output.lower()


@pytest.mark.asyncio
async def test_execute_returns_runtime_error(
        monkeypatch: pytest.MonkeyPatch,
        agent_tool_service: AgentToolService,
        fake_policy_service: FakePolicyService,
) -> None:
    fake_policy_service.responses["should_agent_run_command"] = True

    def _raise_runtime_error(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(subprocess, "run", _raise_runtime_error)
    result = await agent_tool_service.execute("ls")

    assert result.executed is False
    assert "failed" in result.output.lower()


@pytest.mark.asyncio
async def test_execute_captures_non_zero_exit_code(
        tmp_path: Path,
        agent_tool_service: AgentToolService,
        fake_policy_service: FakePolicyService,
) -> None:
    fake_policy_service.responses["should_agent_run_command"] = True

    result = await agent_tool_service.execute("cat nonexistent_file.txt")

    assert result.executed is True
    assert "exit_code: 1" in result.output or "exit_code: 2" in result.output
    assert "no such file" in result.output.lower() or "not found" in result.output.lower()


@pytest.mark.asyncio
async def test_execute_captures_stderr(
        tmp_path: Path,
        agent_tool_service: AgentToolService,
        fake_policy_service: FakePolicyService,
) -> None:
    fake_policy_service.responses["should_agent_run_command"] = True

    result = await agent_tool_service.execute("ls nonexistent_dir")

    assert result.executed is True
    assert "stderr:" in result.output
    assert "no such file" in result.output.lower() or "not found" in result.output.lower()
