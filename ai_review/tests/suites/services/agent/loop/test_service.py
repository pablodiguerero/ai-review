import asyncio

import pytest

from ai_review.services.agent.checkpoint.schema import AgentCheckpointSchema, AgentCheckpointStage
from ai_review.services.agent.loop.schema import AgentAction, AgentStepSchema, AgentTraceSchema
from ai_review.services.agent.loop.service import AgentLoopService, AgentVerificationAborted
from ai_review.services.agent.tool.schema import AgentToolResultSchema
from ai_review.services.llm.types import ChatResultSchema
from ai_review.tests.fixtures.services.agent.checkpoint import FakeAgentCheckpointService
from ai_review.tests.fixtures.services.agent.tool import FakeAgentToolService
from ai_review.tests.fixtures.services.llm import FakeLLMClient
from ai_review.tests.fixtures.services.prompt import FakePromptService


def sequence_chat(outputs: list[str]):
    async def chat(prompt: str, prompt_system: str, json_mode: bool = False) -> ChatResultSchema:
        return ChatResultSchema(text=outputs.pop(0))

    return chat


def sequence_chat_results(outputs: list[ChatResultSchema]):
    async def chat(prompt: str, prompt_system: str, json_mode: bool = False) -> ChatResultSchema:
        return outputs.pop(0)

    return chat


def sequence_tool_results(outputs: list[AgentToolResultSchema]):
    async def execute(command: str) -> AgentToolResultSchema:
        return outputs.pop(0)

    return execute


class FakeClock:
    def __init__(self, start: float = 0.0):
        self.value = start

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


@pytest.mark.asyncio
async def test_run_returns_final_when_llm_returns_final(
        monkeypatch: pytest.MonkeyPatch,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
        fake_agent_tool_service: FakeAgentToolService,
) -> None:
    monkeypatch.setattr(
        fake_llm_client,
        "chat",
        sequence_chat(['{"action":"FINAL","content":"done"}']),
    )

    result = await agent_loop_service.run("PROMPT", "SYSTEM")

    assert result.final_text == "done"
    assert result.stop_reason == "final"
    assert len(result.traces) == 1
    assert fake_agent_tool_service.calls == []


@pytest.mark.asyncio
async def test_run_breaks_to_force_final_when_parse_fails(
        monkeypatch: pytest.MonkeyPatch,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
        fake_agent_tool_service: FakeAgentToolService,
) -> None:
    agent_loop_service.empty_response_retries = 0
    agent_loop_service.force_final_attempts = 1
    monkeypatch.setattr(
        fake_llm_client,
        "chat",
        sequence_chat([
            "not-json",
            '{"action":"FINAL","content":"recovered"}',
        ]),
    )

    result = await agent_loop_service.run("PROMPT", "SYSTEM")

    assert result.stop_reason == "forced_final"
    assert result.final_text == "recovered"
    assert fake_agent_tool_service.calls == []


@pytest.mark.asyncio
async def test_run_aborts_when_the_llm_fails_before_any_verification(
        monkeypatch: pytest.MonkeyPatch,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
) -> None:
    agent_loop_service.empty_response_retries = 0
    agent_loop_service.force_final_attempts = 1
    agent_loop_service.min_tool_calls = 2

    calls: list[int] = []

    async def failing_chat(prompt: str, prompt_system: str, json_mode: bool = False):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("stream transport error")

        return ChatResultSchema(
            text='{"action":"FINAL","content":"LGTM, no issues"}',
            total_tokens=1,
            prompt_tokens=1,
            completion_tokens=1,
        )

    monkeypatch.setattr(fake_llm_client, "chat", failing_chat)

    with pytest.raises(AgentVerificationAborted, match="0/2"):
        await agent_loop_service.run("PROMPT", "SYSTEM")

    assert len(calls) == 1


@pytest.mark.asyncio
async def test_run_breaks_to_force_final_when_the_llm_call_fails(
        monkeypatch: pytest.MonkeyPatch,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
) -> None:
    agent_loop_service.empty_response_retries = 0
    agent_loop_service.force_final_attempts = 1
    agent_loop_service.min_tool_calls = 0

    calls: list[int] = []

    async def failing_chat(prompt: str, prompt_system: str, json_mode: bool = False):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("stream hit the max_tokens limit mid-answer")

        return ChatResultSchema(
            text='{"action":"FINAL","content":"recovered"}',
            total_tokens=1,
            prompt_tokens=1,
            completion_tokens=1,
        )

    monkeypatch.setattr(fake_llm_client, "chat", failing_chat)

    result = await agent_loop_service.run("PROMPT", "SYSTEM")

    assert result.stop_reason == "forced_final"
    assert result.final_text == "recovered"


@pytest.mark.asyncio
async def test_chat_retries_empty_response_then_succeeds(
        monkeypatch: pytest.MonkeyPatch,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
) -> None:
    agent_loop_service.empty_response_retries = 2
    monkeypatch.setattr(
        fake_llm_client,
        "chat",
        sequence_chat([
            "",
            '{"action":"FINAL","content":"after-empty"}',
        ]),
    )

    result = await agent_loop_service.run("PROMPT", "SYSTEM")

    assert result.stop_reason == "final"
    assert result.final_text == "after-empty"


@pytest.mark.asyncio
async def test_run_executes_tool_call_then_returns_final(
        monkeypatch: pytest.MonkeyPatch,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
        fake_agent_tool_service: FakeAgentToolService,
) -> None:
    monkeypatch.setattr(
        fake_llm_client,
        "chat",
        sequence_chat([
            '{"action":"TOOL_CALL","command":"rg foo src"}',
            '{"action":"FINAL","content":"review-complete"}',
        ]),
    )
    fake_agent_tool_service.responses["execute"] = "match-line"

    result = await agent_loop_service.run("PROMPT", "SYSTEM")

    assert result.stop_reason == "final"
    assert result.final_text == "review-complete"
    assert len(result.traces) == 2
    assert fake_agent_tool_service.calls == [("execute", {"command": "rg foo src"})]
    assert result.traces[0].tool_output == "match-line"


@pytest.mark.asyncio
async def test_run_blocks_duplicate_tool_call_signature(
        monkeypatch: pytest.MonkeyPatch,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
        fake_agent_tool_service: FakeAgentToolService,
) -> None:
    monkeypatch.setattr(
        fake_llm_client,
        "chat",
        sequence_chat([
            '{"action":"TOOL_CALL","command":"ls"}',
            '{"action":"TOOL_CALL","command":"ls"}',
            '{"action":"FINAL","content":"done"}',
        ]),
    )

    result = await agent_loop_service.run("PROMPT", "SYSTEM")

    assert result.final_text == "done"
    assert len(fake_agent_tool_service.calls) == 1
    assert "Duplicate tool call blocked" in (result.traces[1].warning or "")


@pytest.mark.asyncio
async def test_run_forces_final_when_context_limit_reached(
        monkeypatch: pytest.MonkeyPatch,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
        fake_prompt_service: FakePromptService,
        fake_agent_tool_service: FakeAgentToolService,
) -> None:
    monkeypatch.setattr(
        fake_llm_client,
        "chat",
        sequence_chat([
            '{"action":"TOOL_CALL","command":"cat big.txt"}',
            '{"action":"FINAL","content":"forced-final"}',
        ]),
    )
    fake_agent_tool_service.responses["execute"] = "0123456789"
    agent_loop_service.max_context_chars = 1

    result = await agent_loop_service.run("PROMPT", "SYSTEM")

    assert result.stop_reason == "forced_final"
    assert result.final_text == "forced-final"
    assert any(
        call[0] == "build_agent_request" and call[1]["force_final"] is True
        for call in fake_prompt_service.calls
    )


@pytest.mark.asyncio
async def test_force_final_skips_when_forced_response_is_never_final(
        monkeypatch: pytest.MonkeyPatch,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
        fake_agent_tool_service: FakeAgentToolService,
) -> None:
    agent_loop_service.empty_response_retries = 0
    agent_loop_service.force_final_attempts = 2
    monkeypatch.setattr(
        fake_llm_client,
        "chat",
        sequence_chat([
            '{"action":"TOOL_CALL","command":"cat big.txt"}',
            '{"action":"TOOL_CALL","command":"cat x"}',
            '{"action":"TOOL_CALL","command":"cat y"}',
        ]),
    )
    fake_agent_tool_service.responses["execute"] = "0123456789"
    agent_loop_service.max_context_chars = 1

    result = await agent_loop_service.run("PROMPT", "SYSTEM")

    assert result.stop_reason == "forced_final_no_review"
    assert result.final_text == ""


@pytest.mark.asyncio
async def test_run_clears_internal_state_between_runs(
        monkeypatch: pytest.MonkeyPatch,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
        fake_agent_tool_service: FakeAgentToolService,
) -> None:
    monkeypatch.setattr(
        fake_llm_client,
        "chat",
        sequence_chat([
            '{"action":"TOOL_CALL","command":"ls"}',
            '{"action":"FINAL","content":"one"}',
        ]),
    )
    await agent_loop_service.run("PROMPT", "SYSTEM")

    monkeypatch.setattr(
        fake_llm_client,
        "chat",
        sequence_chat([
            '{"action":"TOOL_CALL","command":"ls"}',
            '{"action":"FINAL","content":"two"}',
        ]),
    )
    result = await agent_loop_service.run("PROMPT", "SYSTEM")

    assert result.final_text == "two"
    assert fake_agent_tool_service.calls.count(("execute", {"command": "ls"})) == 2


@pytest.mark.asyncio
async def test_run_forces_final_when_max_iterations_reached(
        monkeypatch: pytest.MonkeyPatch,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
        fake_prompt_service: FakePromptService,
        fake_agent_tool_service: FakeAgentToolService,
) -> None:
    monkeypatch.setattr(
        fake_llm_client,
        "chat",
        sequence_chat([
            '{"action":"TOOL_CALL","command":"ls"}',
            '{"action":"TOOL_CALL","command":"cat a.py"}',
            '{"action":"FINAL","content":"forced"}',
        ]),
    )
    agent_loop_service.max_iterations = 2

    result = await agent_loop_service.run("PROMPT", "SYSTEM")

    assert result.stop_reason == "forced_final"
    assert result.final_text == "forced"
    assert any(
        call[0] == "build_agent_request" and call[1]["force_final"] is True
        for call in fake_prompt_service.calls
    )


@pytest.mark.asyncio
async def test_run_handles_coerced_list_content_as_final(
        monkeypatch: pytest.MonkeyPatch,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
) -> None:
    monkeypatch.setattr(
        fake_llm_client,
        "chat",
        sequence_chat(['{"action":"FINAL","content":[]}']),
    )

    result = await agent_loop_service.run("PROMPT", "SYSTEM")

    assert result.stop_reason == "final"
    assert result.final_text == "[]"


@pytest.mark.asyncio
async def test_run_skips_when_llm_only_returns_empty(
        monkeypatch: pytest.MonkeyPatch,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
) -> None:
    agent_loop_service.empty_response_retries = 0
    agent_loop_service.force_final_attempts = 1
    monkeypatch.setattr(
        fake_llm_client,
        "chat",
        sequence_chat(["", ""]),
    )

    result = await agent_loop_service.run("PROMPT", "SYSTEM")

    assert result.stop_reason == "forced_final_no_review"
    assert result.final_text == ""


@pytest.mark.asyncio
async def test_run_persists_llm_tokens_in_traces(
        monkeypatch: pytest.MonkeyPatch,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
) -> None:
    monkeypatch.setattr(
        fake_llm_client,
        "chat",
        sequence_chat_results([
            ChatResultSchema(
                text='{"action":"TOOL_CALL","command":"ls"}',
                total_tokens=30,
                prompt_tokens=10,
                completion_tokens=20,
            ),
            ChatResultSchema(
                text='{"action":"FINAL","content":"done"}',
                total_tokens=15,
                prompt_tokens=5,
                completion_tokens=10,
            ),
        ]),
    )

    result = await agent_loop_service.run("PROMPT", "SYSTEM")

    assert result.final_text == "done"
    assert result.traces[0].prompt_tokens == 10
    assert result.traces[0].completion_tokens == 20
    assert result.traces[1].prompt_tokens == 5
    assert result.traces[1].completion_tokens == 10
    assert result.prompt_tokens == 15
    assert result.completion_tokens == 30
    assert result.total_tokens == 45


@pytest.mark.asyncio
async def test_run_counts_only_executed_tool_calls_not_blocked(
        monkeypatch: pytest.MonkeyPatch,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
        fake_agent_tool_service: FakeAgentToolService,
) -> None:
    monkeypatch.setattr(
        fake_llm_client,
        "chat",
        sequence_chat([
            '{"action":"TOOL_CALL","command":"cat blocked.py"}',
            '{"action":"TOOL_CALL","command":"cat allowed.py"}',
            '{"action":"FINAL","content":"done"}',
        ]),
    )
    monkeypatch.setattr(
        fake_agent_tool_service,
        "execute",
        sequence_tool_results([
            AgentToolResultSchema(command="cat blocked.py", output="blocked", executed=False),
            AgentToolResultSchema(command="cat allowed.py", output="ok", executed=True),
        ]),
    )

    result = await agent_loop_service.run("PROMPT", "SYSTEM")

    assert result.stop_reason == "final"
    assert result.executed_tool_calls == 1
    assert result.blocked_tool_calls == 1
    assert result.iterations == 3


@pytest.mark.asyncio
async def test_run_counts_duplicate_tool_call_as_blocked(
        monkeypatch: pytest.MonkeyPatch,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
        fake_agent_tool_service: FakeAgentToolService,
) -> None:
    monkeypatch.setattr(
        fake_llm_client,
        "chat",
        sequence_chat([
            '{"action":"TOOL_CALL","command":"ls"}',
            '{"action":"TOOL_CALL","command":"ls"}',
            '{"action":"FINAL","content":"done"}',
        ]),
    )

    result = await agent_loop_service.run("PROMPT", "SYSTEM")

    assert result.executed_tool_calls == 1
    assert result.blocked_tool_calls == 1


@pytest.mark.asyncio
async def test_run_min_tool_calls_gating_uses_executed_not_blocked(
        monkeypatch: pytest.MonkeyPatch,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
        fake_agent_tool_service: FakeAgentToolService,
) -> None:
    agent_loop_service.min_tool_calls = 1
    monkeypatch.setattr(
        fake_llm_client,
        "chat",
        sequence_chat([
            '{"action":"TOOL_CALL","command":"cat blocked.py"}',
            '{"action":"FINAL","content":"too-early"}',
            '{"action":"TOOL_CALL","command":"cat allowed.py"}',
            '{"action":"FINAL","content":"done"}',
        ]),
    )
    monkeypatch.setattr(
        fake_agent_tool_service,
        "execute",
        sequence_tool_results([
            AgentToolResultSchema(command="cat blocked.py", output="blocked", executed=False),
            AgentToolResultSchema(command="cat allowed.py", output="ok", executed=True),
        ]),
    )

    result = await agent_loop_service.run("PROMPT", "SYSTEM")

    assert result.stop_reason == "final"
    assert result.final_text == "done"
    assert result.executed_tool_calls == 1
    assert result.blocked_tool_calls == 1


@pytest.mark.asyncio
async def test_run_populates_counters_on_final(
        monkeypatch: pytest.MonkeyPatch,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
        fake_agent_tool_service: FakeAgentToolService,
) -> None:
    monkeypatch.setattr(
        fake_llm_client,
        "chat",
        sequence_chat([
            '{"action":"TOOL_CALL","command":"ls"}',
            '{"action":"FINAL","content":"done"}',
        ]),
    )

    result = await agent_loop_service.run("PROMPT", "SYSTEM")

    assert result.iterations == 2
    assert result.executed_tool_calls == 1
    assert result.blocked_tool_calls == 0


@pytest.mark.asyncio
async def test_run_populates_counters_on_forced_final(
        monkeypatch: pytest.MonkeyPatch,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
        fake_prompt_service: FakePromptService,
        fake_agent_tool_service: FakeAgentToolService,
) -> None:
    monkeypatch.setattr(
        fake_llm_client,
        "chat",
        sequence_chat([
            '{"action":"TOOL_CALL","command":"cat big.txt"}',
            '{"action":"FINAL","content":"forced-final"}',
        ]),
    )
    fake_agent_tool_service.responses["execute"] = "0123456789"
    agent_loop_service.max_context_chars = 1

    result = await agent_loop_service.run("PROMPT", "SYSTEM")

    assert result.stop_reason == "forced_final"
    assert result.iterations == 1
    assert result.executed_tool_calls == 1
    assert result.blocked_tool_calls == 0


@pytest.mark.asyncio
async def test_run_logs_fixed_format_summary_line(
        capsys: pytest.CaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
        fake_agent_tool_service: FakeAgentToolService,
) -> None:
    monkeypatch.setattr(
        fake_llm_client,
        "chat",
        sequence_chat([
            '{"action":"TOOL_CALL","command":"ls"}',
            '{"action":"FINAL","content":"done"}',
        ]),
    )

    await agent_loop_service.run("PROMPT", "SYSTEM")
    output = capsys.readouterr().out

    assert "Agent loop summary: iterations=2 executed_tool_calls=1 blocked_tool_calls=0 stop_reason=final" in output


@pytest.mark.asyncio
async def test_chat_skips_empty_response_retry_once_deadline_passed(
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
) -> None:
    clock = FakeClock()
    agent_loop_service.clock = clock
    agent_loop_service.empty_response_retries = 3

    calls: list[int] = []

    async def chat(prompt: str, prompt_system: str, json_mode: bool = False) -> ChatResultSchema:
        calls.append(1)
        clock.advance(10)
        return ChatResultSchema(text="")

    fake_llm_client.chat = chat

    result = await agent_loop_service._chat("PROMPT", "SYSTEM", deadline_at=5)

    assert result.text == ""
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_run_stops_regular_loop_at_deadline_and_force_finals_once(
        monkeypatch: pytest.MonkeyPatch,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
        fake_agent_tool_service: FakeAgentToolService,
) -> None:
    clock = FakeClock()
    agent_loop_service.clock = clock
    agent_loop_service.deadline_seconds = 5
    agent_loop_service.force_final_attempts = 3

    calls: list[int] = []

    async def chat(prompt: str, prompt_system: str, json_mode: bool = False) -> ChatResultSchema:
        calls.append(1)
        if len(calls) == 1:
            clock.advance(10)
            return ChatResultSchema(text='{"action":"TOOL_CALL","command":"ls"}')
        return ChatResultSchema(text='{"action":"FINAL","content":"done"}')

    monkeypatch.setattr(fake_llm_client, "chat", chat)

    result = await agent_loop_service.run("PROMPT", "SYSTEM")

    assert result.stop_reason == "forced_final"
    assert result.final_text == "done"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_force_final_stops_early_once_past_deadline_budget(
        monkeypatch: pytest.MonkeyPatch,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
        fake_agent_tool_service: FakeAgentToolService,
) -> None:
    clock = FakeClock()
    agent_loop_service.clock = clock
    agent_loop_service.deadline_seconds = 5
    agent_loop_service.force_final_attempts = 3
    agent_loop_service.max_iterations = 1

    calls: list[int] = []

    async def chat(prompt: str, prompt_system: str, json_mode: bool = False) -> ChatResultSchema:
        calls.append(1)
        if len(calls) == 1:
            return ChatResultSchema(text='{"action":"TOOL_CALL","command":"ls"}')
        clock.advance(1000)
        return ChatResultSchema(text='{"action":"TOOL_CALL","command":"cat x"}')

    monkeypatch.setattr(fake_llm_client, "chat", chat)

    result = await agent_loop_service.run("PROMPT", "SYSTEM")

    assert result.stop_reason == "forced_final_no_review"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_run_deadline_none_leaves_loop_unaffected(
        monkeypatch: pytest.MonkeyPatch,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
) -> None:
    agent_loop_service.deadline_seconds = None
    monkeypatch.setattr(
        fake_llm_client,
        "chat",
        sequence_chat(['{"action":"FINAL","content":"done"}']),
    )

    result = await agent_loop_service.run("PROMPT", "SYSTEM")

    assert result.stop_reason == "final"
    assert result.final_text == "done"


@pytest.mark.asyncio
async def test_force_final_refuses_to_publish_when_deadline_bypasses_min_tool_calls(
        monkeypatch: pytest.MonkeyPatch,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
) -> None:
    agent_loop_service.min_tool_calls = 2
    agent_loop_service.deadline_seconds = 1

    clock_values = iter([0.0, 100.0])
    agent_loop_service.clock = lambda: next(clock_values, 100.0)

    calls: list[int] = []

    async def chat(prompt: str, prompt_system: str, json_mode: bool = False) -> ChatResultSchema:
        calls.append(1)
        return ChatResultSchema(text='{"action":"FINAL","content":"should-not-be-posted"}')

    monkeypatch.setattr(fake_llm_client, "chat", chat)

    result = await agent_loop_service.run("PROMPT", "SYSTEM")

    assert result.stop_reason == "forced_final_no_review"
    assert result.final_text == ""
    assert len(calls) == 0


@pytest.mark.asyncio
async def test_run_is_isolated_across_concurrent_executions(
        monkeypatch: pytest.MonkeyPatch,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
        fake_prompt_service: FakePromptService,
) -> None:
    monkeypatch.setattr(fake_prompt_service, "build_agent_request", lambda **kwargs: kwargs["original_prompt"])

    calls_a: list[int] = []
    calls_b: list[int] = []

    async def chat(prompt: str, prompt_system: str, json_mode: bool = False) -> ChatResultSchema:
        await asyncio.sleep(0)
        if prompt == "TASK_A":
            calls_a.append(1)
            if len(calls_a) == 1:
                return ChatResultSchema(text='{"action":"TOOL_CALL","command":"ls a"}')
            return ChatResultSchema(text='{"action":"FINAL","content":"result-a"}')

        calls_b.append(1)
        if len(calls_b) == 1:
            return ChatResultSchema(text='{"action":"TOOL_CALL","command":"ls b"}')
        return ChatResultSchema(text='{"action":"FINAL","content":"result-b"}')

    monkeypatch.setattr(fake_llm_client, "chat", chat)

    result_a, result_b = await asyncio.gather(
        agent_loop_service.run("TASK_A", "SYSTEM"),
        agent_loop_service.run("TASK_B", "SYSTEM"),
    )

    assert result_a.final_text == "result-a"
    assert result_b.final_text == "result-b"
    assert result_a.executed_tool_calls == 1
    assert result_b.executed_tool_calls == 1
    assert result_a.iterations == 2
    assert result_b.iterations == 2
    assert len(result_a.traces) == 2
    assert len(result_b.traces) == 2
    assert result_a.traces[0].step.command == "ls a"
    assert result_b.traces[0].step.command == "ls b"


@pytest.mark.asyncio
async def test_force_final_recovers_when_first_attempt_raises_and_second_succeeds(
        monkeypatch: pytest.MonkeyPatch,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
) -> None:
    agent_loop_service.empty_response_retries = 0
    agent_loop_service.force_final_attempts = 2

    calls: list[int] = []

    async def chat(prompt: str, prompt_system: str, json_mode: bool = False) -> ChatResultSchema:
        calls.append(1)
        if len(calls) == 1:
            return ChatResultSchema(text="not-json")
        if len(calls) == 2:
            raise RuntimeError("stream error")
        return ChatResultSchema(text='{"action":"FINAL","content":"recovered"}')

    monkeypatch.setattr(fake_llm_client, "chat", chat)

    result = await agent_loop_service.run("PROMPT", "SYSTEM")

    assert result.stop_reason == "forced_final"
    assert result.final_text == "recovered"
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_force_final_reraises_last_error_when_every_attempt_fails(
        monkeypatch: pytest.MonkeyPatch,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
) -> None:
    agent_loop_service.empty_response_retries = 0
    agent_loop_service.force_final_attempts = 2

    calls: list[int] = []

    async def chat(prompt: str, prompt_system: str, json_mode: bool = False) -> ChatResultSchema:
        calls.append(1)
        if len(calls) == 1:
            return ChatResultSchema(text="not-json")
        raise RuntimeError(f"boom-{len(calls)}")

    monkeypatch.setattr(fake_llm_client, "chat", chat)

    with pytest.raises(RuntimeError, match="boom-3"):
        await agent_loop_service.run("PROMPT", "SYSTEM")

    assert len(calls) == 3


@pytest.mark.asyncio
async def test_force_final_stops_without_further_calls_when_deadline_exhausted_after_failure(
        monkeypatch: pytest.MonkeyPatch,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
) -> None:
    agent_loop_service.deadline_seconds = 1
    agent_loop_service.llm_request_timeout = 10
    agent_loop_service.force_final_attempts = 3

    clock_values = iter([0.0])
    agent_loop_service.clock = lambda: next(clock_values, 100.0)

    calls: list[int] = []

    async def chat(prompt: str, prompt_system: str, json_mode: bool = False) -> ChatResultSchema:
        calls.append(1)
        raise RuntimeError("stream stalled")

    monkeypatch.setattr(fake_llm_client, "chat", chat)

    with pytest.raises(RuntimeError, match="stream stalled"):
        await agent_loop_service.run("PROMPT", "SYSTEM")

    assert len(calls) == 1


@pytest.mark.asyncio
async def test_run_does_not_touch_checkpoint_when_key_not_provided(
        monkeypatch: pytest.MonkeyPatch,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
        fake_agent_checkpoint_service: FakeAgentCheckpointService,
) -> None:
    monkeypatch.setattr(
        fake_llm_client,
        "chat",
        sequence_chat([
            '{"action":"TOOL_CALL","command":"ls"}',
            '{"action":"FINAL","content":"done"}',
        ]),
    )

    result = await agent_loop_service.run("PROMPT", "SYSTEM")

    assert result.stop_reason == "final"
    assert fake_agent_checkpoint_service.calls == []


@pytest.mark.asyncio
async def test_run_saves_checkpoint_after_each_iteration_and_on_success(
        monkeypatch: pytest.MonkeyPatch,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
        fake_agent_checkpoint_service: FakeAgentCheckpointService,
) -> None:
    monkeypatch.setattr(
        fake_llm_client,
        "chat",
        sequence_chat([
            '{"action":"TOOL_CALL","command":"ls"}',
            '{"action":"FINAL","content":"done"}',
        ]),
    )

    result = await agent_loop_service.run("PROMPT", "SYSTEM", checkpoint_key="k1", checkpoint_head_sha="sha1")

    assert result.stop_reason == "final"
    save_calls = [call for call in fake_agent_checkpoint_service.calls if call[0] == "save"]
    assert len(save_calls) >= 2

    stored = fake_agent_checkpoint_service.store["k1"]
    assert stored.stage == AgentCheckpointStage.REVIEWED
    assert stored.head_sha == "sha1"
    assert stored.round == 0
    assert stored.executed_tool_calls == 1
    assert stored.iterations == 2
    assert all(call[0] != "delete" for call in fake_agent_checkpoint_service.calls)


@pytest.mark.asyncio
async def test_run_investigating_same_commit_continues_from_next_iteration(
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
        fake_agent_checkpoint_service: FakeAgentCheckpointService,
) -> None:
    fake_agent_checkpoint_service.store["k1"] = AgentCheckpointSchema(
        key="k1",
        head_sha="sha1",
        round=0,
        stage=AgentCheckpointStage.INVESTIGATING,
        traces=[],
        executed_tool_calls=1,
        blocked_tool_calls=0,
        iterations=2,
        context_used=10,
        signatures=["ls"],
        created_at="2026-08-19T00:00:00+00:00",
        updated_at="2026-08-19T00:00:00+00:00",
    )

    monkeypatch.setattr(
        fake_llm_client,
        "chat",
        sequence_chat(['{"action":"FINAL","content":"done"}']),
    )

    result = await agent_loop_service.run("PROMPT", "SYSTEM", checkpoint_key="k1", checkpoint_head_sha="sha1")
    output = capsys.readouterr().out

    assert result.stop_reason == "final"
    assert result.iterations == 3
    assert result.executed_tool_calls == 1
    assert "starting round 0 (resumed: continue)" in output

    stored = fake_agent_checkpoint_service.store["k1"]
    assert stored.round == 0


@pytest.mark.asyncio
async def test_run_needs_final_same_commit_checkpoint_skips_straight_to_force_final(
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
        fake_agent_checkpoint_service: FakeAgentCheckpointService,
) -> None:
    fake_agent_checkpoint_service.store["k1"] = AgentCheckpointSchema(
        key="k1",
        head_sha="sha1",
        round=3,
        stage=AgentCheckpointStage.NEEDS_FINAL,
        traces=[],
        executed_tool_calls=2,
        blocked_tool_calls=0,
        iterations=5,
        context_used=10,
        signatures=["ls"],
        created_at="2026-08-19T00:00:00+00:00",
        updated_at="2026-08-19T00:00:00+00:00",
    )

    calls: list[int] = []

    async def chat(prompt: str, prompt_system: str, json_mode: bool = False) -> ChatResultSchema:
        calls.append(1)
        return ChatResultSchema(text='{"action":"FINAL","content":"resumed-final"}')

    monkeypatch.setattr(fake_llm_client, "chat", chat)

    result = await agent_loop_service.run("PROMPT", "SYSTEM", checkpoint_key="k1", checkpoint_head_sha="sha1")
    output = capsys.readouterr().out

    assert result.stop_reason == "forced_final"
    assert result.final_text == "resumed-final"
    assert len(calls) == 1
    assert "starting round 3 (resumed: recover-final)" in output

    stored = fake_agent_checkpoint_service.store["k1"]
    assert stored.stage == AgentCheckpointStage.REVIEWED
    assert stored.round == 3


@pytest.mark.asyncio
async def test_run_persists_checkpoint_after_successful_forced_final_with_clean_traces(
        monkeypatch: pytest.MonkeyPatch,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
        fake_agent_tool_service: FakeAgentToolService,
        fake_agent_checkpoint_service: FakeAgentCheckpointService,
) -> None:
    agent_loop_service.min_tool_calls = 0
    agent_loop_service.max_iterations = 1

    monkeypatch.setattr(
        fake_llm_client,
        "chat",
        sequence_chat([
            '{"action":"TOOL_CALL","command":"ls"}',
            '{"action":"FINAL","content":"forced-done"}',
        ]),
    )

    result = await agent_loop_service.run("PROMPT", "SYSTEM", checkpoint_key="k1")

    assert result.stop_reason == "forced_final"
    stored = fake_agent_checkpoint_service.store["k1"]
    assert stored.stage == AgentCheckpointStage.REVIEWED
    assert len(stored.traces) == 1
    assert stored.traces[0].step.command == "ls"
    assert all(call[0] != "delete" for call in fake_agent_checkpoint_service.calls)


@pytest.mark.asyncio
async def test_second_run_with_needs_final_checkpoint_makes_exactly_one_llm_call_and_publishes(
        monkeypatch: pytest.MonkeyPatch,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
        fake_agent_checkpoint_service: FakeAgentCheckpointService,
) -> None:
    agent_loop_service.max_iterations = 1
    agent_loop_service.force_final_attempts = 2

    monkeypatch.setattr(
        fake_llm_client,
        "chat",
        sequence_chat([
            "not-json",
            '{"action":"TOOL_CALL","command":"cat x"}',
            '{"action":"TOOL_CALL","command":"cat y"}',
        ]),
    )

    first_result = await agent_loop_service.run("PROMPT", "SYSTEM", checkpoint_key="retry-key")

    assert first_result.stop_reason == "forced_final_no_review"
    assert fake_agent_checkpoint_service.store["retry-key"].stage == AgentCheckpointStage.NEEDS_FINAL

    calls: list[int] = []

    async def chat(prompt: str, prompt_system: str, json_mode: bool = False) -> ChatResultSchema:
        calls.append(1)
        return ChatResultSchema(text='{"action":"FINAL","content":"published-on-retry"}')

    monkeypatch.setattr(fake_llm_client, "chat", chat)

    second_result = await agent_loop_service.run("PROMPT", "SYSTEM", checkpoint_key="retry-key")

    assert len(calls) == 1
    assert second_result.stop_reason == "forced_final"
    assert second_result.final_text == "published-on-retry"
    assert fake_agent_checkpoint_service.store["retry-key"].stage == AgentCheckpointStage.REVIEWED


@pytest.mark.asyncio
async def test_run_new_commit_round_folds_prior_traces_into_synopsis_and_resets_signatures(
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
        fake_prompt_service: FakePromptService,
        fake_agent_tool_service: FakeAgentToolService,
        fake_agent_checkpoint_service: FakeAgentCheckpointService,
) -> None:
    agent_loop_service.resume_min_new_tool_calls = 1
    fake_agent_checkpoint_service.store["k1"] = AgentCheckpointSchema(
        key="k1",
        head_sha="old-sha",
        round=0,
        stage=AgentCheckpointStage.REVIEWED,
        traces=[
            AgentTraceSchema(
                step=AgentStepSchema(action=AgentAction.TOOL_CALL, command="ls"),
                iteration=1,
                raw_output='{"action":"TOOL_CALL","command":"ls"}',
                tool_output="old evidence output",
            ),
        ],
        signatures=["ls"],
        executed_tool_calls=1,
        blocked_tool_calls=0,
        iterations=2,
        context_used=5,
        created_at="2026-08-19T00:00:00+00:00",
        updated_at="2026-08-19T00:00:00+00:00",
    )

    monkeypatch.setattr(
        fake_llm_client,
        "chat",
        sequence_chat([
            '{"action":"TOOL_CALL","command":"ls"}',
            '{"action":"FINAL","content":"new-round-done"}',
        ]),
    )

    result = await agent_loop_service.run("PROMPT", "SYSTEM", checkpoint_key="k1", checkpoint_head_sha="new-sha")
    output = capsys.readouterr().out

    assert result.stop_reason == "final"
    assert result.final_text == "new-round-done"
    assert fake_agent_tool_service.calls == [("execute", {"command": "ls"})]
    assert "starting round 1 (resumed: new-commit)" in output
    assert len(result.traces) == 2
    assert all(trace.tool_output != "old evidence output" for trace in result.traces)

    build_calls = [call for call in fake_prompt_service.calls if call[0] == "build_agent_request"]
    assert "old evidence output" in build_calls[0][1]["prior_synopsis"]

    stored = fake_agent_checkpoint_service.store["k1"]
    assert stored.stage == AgentCheckpointStage.REVIEWED
    assert stored.round == 1
    assert stored.head_sha == "new-sha"


@pytest.mark.asyncio
async def test_run_same_commit_reviewed_restart_keeps_signatures_and_requires_new_tool_calls_before_final(
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
        fake_agent_tool_service: FakeAgentToolService,
        fake_agent_checkpoint_service: FakeAgentCheckpointService,
) -> None:
    agent_loop_service.resume_min_new_tool_calls = 1
    fake_agent_checkpoint_service.store["k1"] = AgentCheckpointSchema(
        key="k1",
        head_sha="sha1",
        round=0,
        stage=AgentCheckpointStage.REVIEWED,
        traces=[
            AgentTraceSchema(
                step=AgentStepSchema(action=AgentAction.TOOL_CALL, command="ls"),
                iteration=1,
                raw_output='{"action":"TOOL_CALL","command":"ls"}',
                tool_output="old evidence",
            ),
        ],
        signatures=["ls"],
        executed_tool_calls=1,
        blocked_tool_calls=0,
        iterations=2,
        context_used=5,
        created_at="2026-08-19T00:00:00+00:00",
        updated_at="2026-08-19T00:00:00+00:00",
    )

    monkeypatch.setattr(
        fake_llm_client,
        "chat",
        sequence_chat([
            '{"action":"FINAL","content":"too-early"}',
            '{"action":"TOOL_CALL","command":"cat new.py"}',
            '{"action":"FINAL","content":"done-after-verifying"}',
        ]),
    )
    fake_agent_tool_service.responses["execute"] = "new output"

    result = await agent_loop_service.run("PROMPT", "SYSTEM", checkpoint_key="k1", checkpoint_head_sha="sha1")
    output = capsys.readouterr().out

    assert result.stop_reason == "final"
    assert result.final_text == "done-after-verifying"
    assert fake_agent_tool_service.calls == [("execute", {"command": "cat new.py"})]
    assert any("REJECTED" in (trace.warning or "") for trace in result.traces)
    assert "starting round 1 (resumed: restart)" in output

    stored = fake_agent_checkpoint_service.store["k1"]
    assert stored.round == 1
    assert "ls" in stored.signatures
    assert "cat new.py" in stored.signatures


@pytest.mark.asyncio
async def test_two_run_sequence_new_commit_triggers_genuine_new_round_not_bare_force_final(
        monkeypatch: pytest.MonkeyPatch,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
        fake_agent_tool_service: FakeAgentToolService,
        fake_agent_checkpoint_service: FakeAgentCheckpointService,
) -> None:
    agent_loop_service.resume_min_new_tool_calls = 1

    monkeypatch.setattr(
        fake_llm_client,
        "chat",
        sequence_chat([
            '{"action":"TOOL_CALL","command":"ls"}',
            '{"action":"FINAL","content":"round-one-done"}',
        ]),
    )
    fake_agent_tool_service.responses["execute"] = "round one output"

    first_result = await agent_loop_service.run(
        "PROMPT", "SYSTEM", checkpoint_key="k1", checkpoint_head_sha="sha1"
    )

    assert first_result.stop_reason == "final"
    stored_after_round_one = fake_agent_checkpoint_service.store["k1"]
    assert stored_after_round_one.stage == AgentCheckpointStage.REVIEWED
    assert stored_after_round_one.round == 0
    assert stored_after_round_one.head_sha == "sha1"

    monkeypatch.setattr(
        fake_llm_client,
        "chat",
        sequence_chat([
            '{"action":"TOOL_CALL","command":"cat changed.py"}',
            '{"action":"FINAL","content":"round-two-done"}',
        ]),
    )
    fake_agent_tool_service.responses["execute"] = "round two output"

    second_result = await agent_loop_service.run(
        "PROMPT", "SYSTEM", checkpoint_key="k1", checkpoint_head_sha="sha2"
    )

    assert second_result.stop_reason == "final"
    assert second_result.final_text == "round-two-done"
    assert fake_agent_tool_service.calls == [
        ("execute", {"command": "ls"}),
        ("execute", {"command": "cat changed.py"}),
    ]

    stored_after_round_two = fake_agent_checkpoint_service.store["k1"]
    assert stored_after_round_two.round == 1
    assert stored_after_round_two.head_sha == "sha2"
    assert "round one output" in stored_after_round_two.prior_synopsis
