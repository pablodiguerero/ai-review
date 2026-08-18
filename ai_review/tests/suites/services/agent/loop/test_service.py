import pytest

from ai_review.services.agent.loop.service import AgentLoopService, AgentVerificationAborted
from ai_review.services.llm.types import ChatResultSchema
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

    # An unparseable step must not be dumped as the review; the loop force-finalizes.
    assert result.stop_reason == "forced_final"
    assert result.final_text == "recovered"
    assert fake_agent_tool_service.calls == []


@pytest.mark.asyncio
async def test_run_aborts_when_the_llm_fails_before_any_verification(
        monkeypatch: pytest.MonkeyPatch,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
) -> None:
    # Force-finalizing here would publish a review that ran no verification at
    # all, which is exactly what min_tool_calls exists to prevent.
    agent_loop_service.empty_response_retries = 0
    agent_loop_service.force_final_attempts = 1
    agent_loop_service.min_tool_calls = 2

    calls: list[int] = []

    async def failing_chat(prompt: str, prompt_system: str, json_mode: bool = False):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("stream transport error")

        # Would be published as the review if the loop force-finalized here.
        return ChatResultSchema(
            text='{"action":"FINAL","content":"LGTM, no issues"}',
            total_tokens=1,
            prompt_tokens=1,
            completion_tokens=1,
        )

    monkeypatch.setattr(fake_llm_client, "chat", failing_chat)

    with pytest.raises(AgentVerificationAborted, match="0/2"):
        await agent_loop_service.run("PROMPT", "SYSTEM")

    # One call only: no force-final attempt was made behind the abort.
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_run_breaks_to_force_final_when_the_llm_call_fails(
        monkeypatch: pytest.MonkeyPatch,
        agent_loop_service: AgentLoopService,
        fake_llm_client: FakeLLMClient,
) -> None:
    # A failed intermediate call must keep the traces and force-finalize, not
    # abort the review and throw away everything already paid for.
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

    # A forced response that is a TOOL_CALL (not FINAL) must never be posted as the
    # review — the loop returns empty text so the runner skips the comment.
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

    # Persistent empty content must not post a comment.
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
