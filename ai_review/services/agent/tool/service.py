import shlex
import subprocess
from pathlib import Path

from ai_review.config import settings
from ai_review.libs.logger import get_logger
from ai_review.libs.text import truncate_text
from ai_review.services.agent.tool.schema import AgentToolResultSchema
from ai_review.services.agent.tool.types import AgentToolServiceProtocol
from ai_review.services.policy.types import PolicyServiceProtocol

logger = get_logger("AGENT_TOOL_SERVICE")

SHELL_OPERATORS_BLOCKED_HINT = (
    "Agent command blocked by policy: shell operators are not supported, run one plain command "
    "per TOOL_CALL (no pipes, no &&, no redirects): {command}"
)
TRUNCATION_HINT = (
    "\nOutput truncated: read the file in ranges with sed -n 'A,Bp' FILE or head/tail, "
    "or narrow the search with rg."
)


def _allowed_commands_hint() -> str:
    return ", ".join(pattern.pattern for pattern in settings.agent.allow_commands)


class AgentToolService(AgentToolServiceProtocol):
    def __init__(
            self,
            policy: PolicyServiceProtocol,
            repo_dir: Path = Path(".")
    ):
        self.policy = policy
        self.repo_root = repo_dir.resolve()

        self.command_timeout = settings.agent.command_timeout
        self.max_command_output_chars = settings.agent.max_command_output_chars

    async def execute(self, command: str) -> AgentToolResultSchema:
        command = (command or "").strip()
        command_preview = f"{self.repo_root}#{command}"

        if not command:
            logger.warning("Agent command rejected: empty command")
            return AgentToolResultSchema(
                command=command,
                output="Agent command rejected: empty command",
                executed=False,
            )

        if not self.policy.should_agent_run_command(command):
            if self.policy.command_has_shell_operators(command):
                message = SHELL_OPERATORS_BLOCKED_HINT.format(command=command)
            else:
                message = f"Agent command blocked by policy: {command}. Allowed: {_allowed_commands_hint()}"
            logger.warning(message)
            return AgentToolResultSchema(command=command, output=message, executed=False)

        try:
            argv = shlex.split(command)
        except ValueError as error:
            message = f"Agent command parse error: {command} | {error}"
            logger.warning(message)
            return AgentToolResultSchema(command=command, output=message, executed=False)
        if not argv:
            message = f"Agent command rejected after parsing: {command}"
            logger.warning(message)
            return AgentToolResultSchema(command=command, output=message, executed=False)

        logger.debug(f"Running agent command: {command_preview}, timeout={self.command_timeout}s")
        try:
            result = subprocess.run(
                argv,
                cwd=self.repo_root,
                check=False,
                errors="replace",
                timeout=self.command_timeout,
                encoding="utf-8",
                capture_output=True,
            )
        except subprocess.TimeoutExpired:
            message = f"Agent command timeout: {command_preview}, timeout={self.command_timeout}s"
            logger.warning(message)
            return AgentToolResultSchema(command=command, output=message, executed=False)
        except Exception as error:
            message = f"Agent command failed: {command_preview}:{error}"
            logger.exception(message)
            return AgentToolResultSchema(command=command, output=message, executed=False)

        stdout = result.stdout or ""
        stderr = result.stderr or ""
        logger.debug(
            f"Agent command finished: {command_preview}, exit_code={result.returncode}, "
            f"stdout_chars={len(stdout)}, stderr_chars={len(stderr)}"
        )

        output = (
            f"command: {command}\n"
            f"exit_code: {result.returncode}\n"
            f"stdout: {stdout}\n"
            f"stderr: {stderr}"
        )
        truncated = len(output) > self.max_command_output_chars
        if truncated:
            logger.debug(
                "Agent command output truncated: "
                f"{command}, payload_chars={len(output)}, limit={self.max_command_output_chars}"
            )
            content_budget = max(self.max_command_output_chars - len(TRUNCATION_HINT), 0)
            output = (output[:content_budget] + TRUNCATION_HINT)[:self.max_command_output_chars]
        else:
            output = truncate_text(text=output, limit=self.max_command_output_chars)

        return AgentToolResultSchema(command=command, output=output, executed=True)
