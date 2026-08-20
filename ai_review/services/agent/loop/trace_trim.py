import re

from ai_review.services.agent.loop.schema import AgentTraceSchema

_WHITESPACE_RE = re.compile(r"\s+")


def _collapse(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text or "").strip()


def trim_traces(traces: list[AgentTraceSchema], limit: int) -> tuple[list[AgentTraceSchema], str]:
    if limit > 0 and len(traces) <= limit:
        return list(traces), ""

    if limit <= 0:
        dropped, kept = list(traces), []
    else:
        dropped, kept = traces[:-limit], traces[-limit:]

    lines = [
        f"- round evidence: {trace.step.command} -> {_collapse(trace.tool_output)[:200]}"
        for trace in dropped
        if (trace.tool_output or "").strip()
    ]
    return list(kept), "\n".join(lines)


def combine_synopsis(old: str, new: str) -> str:
    if not new:
        return old
    if not old:
        return new
    return f"{old}\n{new}"
