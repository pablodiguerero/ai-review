from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_review.libs.config.agent import AgentConfig


def test_agent_config_defaults() -> None:
    config = AgentConfig()
    assert config.enabled is False
    assert config.max_iterations == 25
    assert config.max_total_context_chars == 40_000
    assert config.command_timeout == 10
    assert config.max_command_output_chars == 8_000
    assert config.fallback_to_direct_chat is False
    assert config.deadline_seconds is None
    assert config.checkpoint_dir is None
    assert len(config.allow_commands) > 0


def test_agent_config_checkpoint_dir_can_be_set() -> None:
    config = AgentConfig(checkpoint_dir=Path("./.cache/agent"))
    assert config.checkpoint_dir == Path("./.cache/agent")


def test_agent_config_rejects_invalid_limits() -> None:
    with pytest.raises(ValidationError):
        AgentConfig(max_iterations=0)

    with pytest.raises(ValidationError):
        AgentConfig(command_timeout=0)

    with pytest.raises(ValidationError):
        AgentConfig(deadline_seconds=0)


def test_agent_config_default_allow_commands_patterns_are_stable() -> None:
    config = AgentConfig()
    patterns = [pattern.pattern for pattern in config.allow_commands]
    assert patterns == [
        r"^ls(?:\s+.*)?$",
        r"^cat(?:\s+.*)?$",
        r"^head(?:\s+.*)?$",
        r"^tail(?:\s+.*)?$",
        r"^wc(?:\s+.*)?$",
        r"""^sed\s+-n\s+(['"]?)\d+(?:,\d+)?p\1\s+(?!-)\S+$""",
        r"^rg(?![\s\S]*(?:--pre\b|--pre-glob|--hostname-bin|--search-zip|\s-z(?:\s|$)))(?:\s+[\s\S]*)?$",
        r"^grep(?:\s+.*)?$",
        r"^find(?![\s\S]*(?:-exec|-execdir|-ok|-okdir|-delete|-fprint|-fprintf|-fls))(?:\s+[\s\S]*)?$",
        r"^git(?![\s\S]*--output)\s+(?:status|show|diff|log|rev-parse|ls-files)(?:\s+[\s\S]*)?$",
    ]


def test_agent_config_default_allow_commands_match_expected_commands() -> None:
    config = AgentConfig()
    allowlist = config.allow_commands

    def is_allowed(command: str) -> bool:
        return any(pattern.fullmatch(command) for pattern in allowlist)

    assert is_allowed("ls")
    assert is_allowed("ls -la")
    assert is_allowed("cat README.md")
    assert is_allowed("head -n 20 file.py")
    assert is_allowed("tail -n 20 file.py")
    assert is_allowed("wc -l file.py")
    assert is_allowed("sed -n '10,20p' file.py")
    assert is_allowed('sed -n "10,20p" file.py')
    assert is_allowed("sed -n 10,20p file.py")
    assert is_allowed("rg TODO ai_review")
    assert is_allowed("rg -n 'a|b' src")
    assert is_allowed("grep -R foo .")
    assert is_allowed("find . -name '*.py'")
    assert is_allowed("find . -type f -newer x")
    assert is_allowed("git status")
    assert is_allowed("git diff --name-only")
    assert is_allowed("git diff HEAD")
    assert is_allowed("git rev-parse HEAD")

    assert not is_allowed("python -c 'print(1)'")
    assert not is_allowed("git checkout main")
    assert not is_allowed("find . -exec rm {} \\;")
    assert not is_allowed("find\n. -exec rm {} +")
    assert not is_allowed("find \n. -delete")
    assert not is_allowed("sed -i 's/a/b/' file.py")
    assert not is_allowed("sed -n '1,2p' -i")
    assert not is_allowed("rg --pre sh x .")
    assert not is_allowed("rg --pre-glob '*.gz' x .")
    assert not is_allowed("rg --hostname-bin foo x")
    assert not is_allowed("rg --search-zip x")
    assert not is_allowed("rg -z x")
    assert not is_allowed("git diff --output=x")
    assert not is_allowed("git log --output=x")
    assert not is_allowed("")


def test_agent_config_deadline_seconds_accepts_positive_values() -> None:
    config = AgentConfig(deadline_seconds=30)
    assert config.deadline_seconds == 30
