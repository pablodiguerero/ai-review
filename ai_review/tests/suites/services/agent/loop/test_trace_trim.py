from ai_review.services.agent.loop.schema import AgentAction, AgentStepSchema, AgentTraceSchema
from ai_review.services.agent.loop.trace_trim import combine_synopsis, trim_traces


def make_trace(iteration: int, command: str, tool_output: str | None = None, warning: str | None = None):
    return AgentTraceSchema(
        step=AgentStepSchema(action=AgentAction.TOOL_CALL, command=command),
        iteration=iteration,
        raw_output=f'{{"action":"TOOL_CALL","command":"{command}"}}',
        tool_output=tool_output,
        warning=warning,
    )


def test_trim_traces_returns_all_traces_unchanged_when_under_the_limit() -> None:
    traces = [make_trace(1, "ls", "file.py")]

    kept, synopsis = trim_traces(traces, limit=5)

    assert kept == traces
    assert synopsis == ""


def test_trim_traces_keeps_only_the_last_n_traces() -> None:
    traces = [make_trace(i, f"cmd{i}", f"output{i}") for i in range(1, 6)]

    kept, synopsis = trim_traces(traces, limit=2)

    assert kept == traces[-2:]
    assert "cmd1" in synopsis
    assert "cmd2" in synopsis
    assert "cmd3" in synopsis
    assert "cmd4" not in synopsis
    assert "cmd5" not in synopsis


def test_trim_traces_synopsis_line_format_includes_command_and_collapsed_output() -> None:
    traces = [
        make_trace(1, "grep foo src", "line one\n\nline   two"),
        make_trace(2, "cmd2", "output2"),
    ]

    kept, synopsis = trim_traces(traces, limit=1)

    assert kept == traces[-1:]
    assert synopsis == "- round evidence: grep foo src -> line one line two"


def test_trim_traces_truncates_tool_output_to_200_chars() -> None:
    long_output = "x" * 500
    traces = [make_trace(1, "cat big.txt", long_output), make_trace(2, "cmd2", "output2")]

    kept, synopsis = trim_traces(traces, limit=1)

    assert synopsis == f"- round evidence: cat big.txt -> {'x' * 200}"


def test_trim_traces_skips_warnings_only_traces_without_tool_output() -> None:
    traces = [
        make_trace(1, "ls", tool_output=None, warning="Duplicate tool call blocked: ls"),
        make_trace(2, "cmd2", "output2"),
    ]

    kept, synopsis = trim_traces(traces, limit=1)

    assert kept == traces[-1:]
    assert synopsis == ""


def test_trim_traces_with_limit_zero_drops_everything_into_synopsis() -> None:
    traces = [make_trace(1, "ls", "file.py"), make_trace(2, "cat a.py", "content")]

    kept, synopsis = trim_traces(traces, limit=0)

    assert kept == []
    assert "ls" in synopsis
    assert "cat a.py" in synopsis


def test_combine_synopsis_joins_non_empty_pieces_with_a_newline() -> None:
    assert combine_synopsis("old", "new") == "old\nnew"


def test_combine_synopsis_returns_old_when_new_is_empty() -> None:
    assert combine_synopsis("old", "") == "old"


def test_combine_synopsis_returns_new_when_old_is_empty() -> None:
    assert combine_synopsis("", "new") == "new"


def test_combine_synopsis_returns_empty_when_both_are_empty() -> None:
    assert combine_synopsis("", "") == ""
