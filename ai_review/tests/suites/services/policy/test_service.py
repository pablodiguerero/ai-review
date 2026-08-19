import re

import pytest

from ai_review.config import settings
from ai_review.services.policy.service import PolicyService


@pytest.fixture(autouse=True)
def reset_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings.review, "ignore_changes", [])
    monkeypatch.setattr(settings.review, "allow_changes", [])
    monkeypatch.setattr(settings.review, "max_inline_comments", None)
    monkeypatch.setattr(settings.review, "max_context_comments", None)


def test_should_review_skips_if_matches_ignore(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings.review, "ignore_changes", ["*.md"])
    assert not PolicyService.should_review_file("README.md")
    assert PolicyService.should_review_file("main.py")


def test_should_review_allows_if_no_allow_rules(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings.review, "ignore_changes", [])
    monkeypatch.setattr(settings.review, "allow_changes", [])
    assert PolicyService.should_review_file("file.py")


def test_should_review_allows_if_matches_allow(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings.review, "allow_changes", ["src/*.py"])
    assert PolicyService.should_review_file("src/main.py")
    assert not PolicyService.should_review_file("tests/test_main.py")


def test_should_review_skips_if_not_in_allow(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings.review, "allow_changes", ["only/*.py"])
    assert not PolicyService.should_review_file("other/file.py")


def test_ignore_has_priority_over_allow(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings.review, "ignore_changes", ["*.py"])
    monkeypatch.setattr(settings.review, "allow_changes", ["*.py"])
    assert not PolicyService.should_review_file("main.py")


def test_should_agent_run_command_blocks_empty_values() -> None:
    assert not PolicyService.should_agent_run_command("")
    assert not PolicyService.should_agent_run_command("   ")


def test_should_agent_run_command_allows_by_pattern(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.agent, "allow_commands", [re.compile(r"^ls(?:\s+.*)?$")])
    assert PolicyService.should_agent_run_command("ls")
    assert PolicyService.should_agent_run_command("ls -la")


def test_should_agent_run_command_blocks_non_matching_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.agent, "allow_commands", [re.compile(r"^ls(?:\s+.*)?$")])
    assert not PolicyService.should_agent_run_command("cat README.md")


def test_should_agent_run_command_trims_spaces_before_match(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.agent, "allow_commands", [re.compile(r"^git\s+status$")])
    assert PolicyService.should_agent_run_command("   git status   ")


@pytest.mark.parametrize(
    "operator_command",
    [
        "ls && cat foo",
        "ls || cat foo",
        "ls | grep foo",
        "ls; cat foo",
        "cat $(whoami)",
        "cat a.py > b.py",
        "cat a.py < b.py",
        "git status &",
    ],
)
def test_should_agent_run_command_blocks_shell_operators(
        monkeypatch: pytest.MonkeyPatch,
        operator_command: str,
) -> None:
    monkeypatch.setattr(settings.agent, "allow_commands", [re.compile(r"^.*$")])
    assert not PolicyService.should_agent_run_command(operator_command)


@pytest.mark.parametrize(
    "quoted_command",
    [
        'rg -n "a|b" src/',
        "grep -E 'foo|bar' x.py",
        "sed -n '10,20p' file",
        'cat "a && b"',
        "echo `whoami`",
    ],
)
def test_should_agent_run_command_allows_operators_inside_quotes(
        monkeypatch: pytest.MonkeyPatch,
        quoted_command: str,
) -> None:
    monkeypatch.setattr(settings.agent, "allow_commands", [re.compile(r"^.*$")])
    assert PolicyService.should_agent_run_command(quoted_command)


def test_command_has_shell_operators_detects_unquoted_operators() -> None:
    assert PolicyService.command_has_shell_operators("ls && cat foo")
    assert PolicyService.command_has_shell_operators("ls | wc -l")
    assert PolicyService.command_has_shell_operators("cat a && cat b")
    assert PolicyService.command_has_shell_operators("rg x > out.txt")
    assert PolicyService.command_has_shell_operators("echo $(cat x)")
    assert not PolicyService.command_has_shell_operators("ls -la")


def test_command_has_shell_operators_ignores_quoted_operators() -> None:
    assert not PolicyService.command_has_shell_operators('rg -n "a|b" src/')
    assert not PolicyService.command_has_shell_operators("grep -E 'foo|bar' x.py")
    assert not PolicyService.command_has_shell_operators("sed -n '10,20p' file")
    assert not PolicyService.command_has_shell_operators('cat "a && b"')


def test_command_has_shell_operators_returns_false_on_unbalanced_quotes() -> None:
    assert not PolicyService.command_has_shell_operators('"unterminated')


@pytest.mark.parametrize(
    "quoted_operator_command",
    [
        "rg '&&' src",
        "git log --grep '||'",
        "rg -n '<>' src",
    ],
)
def test_should_agent_run_command_allows_quoted_operator_only_arguments(
        quoted_operator_command: str,
) -> None:
    assert not PolicyService.command_has_shell_operators(quoted_operator_command)
    assert PolicyService.should_agent_run_command(quoted_operator_command)


@pytest.mark.parametrize(
    "newline_command",
    [
        "find\n. -exec rm {} +",
        "find \n. -delete",
        "git\ndiff",
    ],
)
def test_should_agent_run_command_blocks_commands_with_newlines(
        monkeypatch: pytest.MonkeyPatch,
        newline_command: str,
) -> None:
    monkeypatch.setattr(settings.agent, "allow_commands", [re.compile(r"^[\s\S]*$")])
    assert not PolicyService.should_agent_run_command(newline_command)


def test_should_agent_run_command_allows_regex_alternation_in_rg_and_grep() -> None:
    assert PolicyService.should_agent_run_command('rg -n "a|b" src/')
    assert PolicyService.should_agent_run_command("grep -E 'foo|bar' x.py")


def test_should_agent_run_command_blocks_command_chaining_and_redirects() -> None:
    assert not PolicyService.should_agent_run_command("ls | wc -l")
    assert not PolicyService.should_agent_run_command("cat a && cat b")
    assert not PolicyService.should_agent_run_command("rg x > out.txt")


def test_should_agent_run_command_blocks_command_substitution_even_without_allowlist_match() -> None:
    assert not PolicyService.should_agent_run_command("echo $(cat x)")


def test_should_agent_run_command_default_patterns_accept_new_read_only_commands() -> None:
    assert PolicyService.should_agent_run_command("head file.py")
    assert PolicyService.should_agent_run_command("head -n 20 file.py")
    assert PolicyService.should_agent_run_command("tail -n 5 file.py")
    assert PolicyService.should_agent_run_command("wc -l file.py")
    assert PolicyService.should_agent_run_command("sed -n '10,20p' x.py")
    assert PolicyService.should_agent_run_command("sed -n 10,20p x.py")
    assert PolicyService.should_agent_run_command("sed -n '5p' x.py")
    assert PolicyService.should_agent_run_command("find . -name '*.py'")
    assert PolicyService.should_agent_run_command("find . -type f -newer x")


def test_should_agent_run_command_rejects_find_with_side_effects() -> None:
    assert not PolicyService.should_agent_run_command("find . -exec rm {} \\;")
    assert not PolicyService.should_agent_run_command("find . -execdir rm {} \\;")
    assert not PolicyService.should_agent_run_command("find . -delete")
    assert not PolicyService.should_agent_run_command("find . -ok rm {} \\;")
    assert not PolicyService.should_agent_run_command("find . -okdir rm {} \\;")
    assert not PolicyService.should_agent_run_command("find . -fprint out.txt")
    assert not PolicyService.should_agent_run_command("find . -fprintf out.txt '%p\\n'")
    assert not PolicyService.should_agent_run_command("find . -fls out.txt")


def test_should_agent_run_command_rejects_sed_in_place_edit() -> None:
    assert not PolicyService.should_agent_run_command("sed -i 's/a/b/' x.py")


def test_should_agent_run_command_rejects_rg_command_execution_flags() -> None:
    assert not PolicyService.should_agent_run_command("rg --pre sh x .")
    assert not PolicyService.should_agent_run_command("rg --pre-glob '*.gz' x .")
    assert not PolicyService.should_agent_run_command("rg --hostname-bin foo x")
    assert not PolicyService.should_agent_run_command("rg --search-zip x")
    assert not PolicyService.should_agent_run_command("rg -z x")
    assert PolicyService.should_agent_run_command("rg -n 'a|b' src")


def test_should_agent_run_command_rejects_git_output_redirect() -> None:
    assert not PolicyService.should_agent_run_command("git diff --output=x")
    assert not PolicyService.should_agent_run_command("git log --output=x")
    assert PolicyService.should_agent_run_command("git diff HEAD")


@pytest.mark.parametrize(
    "sensitive_path_command",
    [
        "cat /proc/self/environ",
        'cat "/proc/self/environ"',
        "cat '/proc/self/environ'",
        "cat /dev/null",
        "head /proc/1/maps",
    ],
)
def test_should_agent_run_command_rejects_proc_and_dev_paths(sensitive_path_command: str) -> None:
    assert not PolicyService.should_agent_run_command(sensitive_path_command)


def test_apply_for_files_filters(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings.review, "ignore_changes", ["*.md"])
    monkeypatch.setattr(settings.review, "allow_changes", ["src/*.py"])

    files = ["README.md", "src/main.py", "tests/test_main.py"]
    allowed = PolicyService.apply_for_files(files)

    assert allowed == ["src/main.py"]


def test_apply_for_inline_comments_with_limit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings.review, "max_inline_comments", 2)
    comments = ["c1", "c2", "c3"]
    limited = PolicyService.apply_for_inline_comments(comments)
    assert limited == ["c1", "c2"]


def test_apply_for_inline_comments_without_limit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings.review, "max_inline_comments", None)
    comments = ["c1", "c2", "c3"]
    limited = PolicyService.apply_for_inline_comments(comments)
    assert limited == comments


def test_apply_for_inline_comments_when_fewer_than_limit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings.review, "max_inline_comments", 5)
    comments = ["c1", "c2"]
    limited = PolicyService.apply_for_inline_comments(comments)
    assert limited == comments


def test_apply_for_context_comments_with_limit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings.review, "max_context_comments", 1)
    comments = ["c1", "c2"]
    limited = PolicyService.apply_for_context_comments(comments)
    assert limited == ["c1"]


def test_apply_for_context_comments_without_limit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings.review, "max_context_comments", None)
    comments = ["c1", "c2", "c3"]
    limited = PolicyService.apply_for_context_comments(comments)
    assert limited == comments
