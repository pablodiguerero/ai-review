from pathlib import Path

import pytest

from ai_review.services.agent.checkpoint.schema import AgentCheckpointSchema
from ai_review.services.agent.checkpoint.service import AgentCheckpointService
from ai_review.services.agent.loop.schema import AgentAction, AgentStepSchema, AgentTraceSchema


def make_checkpoint(key: str) -> AgentCheckpointSchema:
    return AgentCheckpointSchema(
        key=key,
        traces=[],
        executed_tool_calls=2,
        blocked_tool_calls=1,
        iterations=3,
        context_used=42,
        signatures=["ls", "cat a.py"],
        finished_iterations=False,
        created_at="2026-08-19T00:00:00+00:00",
    )


@pytest.mark.asyncio
async def test_save_and_load_round_trip(agent_checkpoint_service: AgentCheckpointService) -> None:
    checkpoint = make_checkpoint("key-1")

    await agent_checkpoint_service.save("key-1", checkpoint)
    loaded = await agent_checkpoint_service.load("key-1")

    assert loaded == checkpoint


@pytest.mark.asyncio
async def test_save_and_load_round_trip_with_tool_call_trace(
        agent_checkpoint_service: AgentCheckpointService,
) -> None:
    checkpoint = AgentCheckpointSchema(
        key="key-1",
        traces=[
            AgentTraceSchema(
                step=AgentStepSchema(action=AgentAction.TOOL_CALL, command="ls"),
                iteration=1,
                raw_output='{"action":"TOOL_CALL","command":"ls"}',
                tool_output="file.py",
            ),
        ],
        executed_tool_calls=1,
        blocked_tool_calls=0,
        iterations=1,
        context_used=7,
        signatures=["ls"],
        finished_iterations=False,
        created_at="2026-08-19T00:00:00+00:00",
    )

    await agent_checkpoint_service.save("key-1", checkpoint)
    loaded = await agent_checkpoint_service.load("key-1")

    assert loaded == checkpoint


@pytest.mark.asyncio
async def test_load_returns_none_when_no_checkpoint_saved(
        agent_checkpoint_service: AgentCheckpointService,
) -> None:
    assert await agent_checkpoint_service.load("missing-key") is None


@pytest.mark.asyncio
async def test_load_returns_none_when_checkpoint_dir_is_none(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ai_review.config.settings.agent.checkpoint_dir", None)
    service = AgentCheckpointService()

    assert await service.load("any-key") is None


@pytest.mark.asyncio
async def test_save_is_noop_when_checkpoint_dir_is_none(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
) -> None:
    monkeypatch.setattr("ai_review.config.settings.agent.checkpoint_dir", None)
    service = AgentCheckpointService()

    await service.save("any-key", make_checkpoint("any-key"))

    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_save_creates_checkpoint_dir_lazily(agent_checkpoint_service: AgentCheckpointService) -> None:
    assert not agent_checkpoint_service.checkpoint_dir.exists()

    await agent_checkpoint_service.save("key-1", make_checkpoint("key-1"))

    assert agent_checkpoint_service.checkpoint_dir.exists()
    assert len(list(agent_checkpoint_service.checkpoint_dir.iterdir())) == 1


@pytest.mark.asyncio
async def test_load_ignores_corrupt_checkpoint_file(agent_checkpoint_service: AgentCheckpointService) -> None:
    agent_checkpoint_service.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = agent_checkpoint_service._path_for_key("key-1")
    path.write_text("not valid json {{{", encoding="utf-8")

    assert await agent_checkpoint_service.load("key-1") is None


@pytest.mark.asyncio
async def test_save_overwrites_a_corrupt_checkpoint_file(agent_checkpoint_service: AgentCheckpointService) -> None:
    agent_checkpoint_service.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = agent_checkpoint_service._path_for_key("key-1")
    path.write_text("not valid json {{{", encoding="utf-8")

    checkpoint = make_checkpoint("key-1")
    await agent_checkpoint_service.save("key-1", checkpoint)
    loaded = await agent_checkpoint_service.load("key-1")

    assert loaded == checkpoint


@pytest.mark.asyncio
async def test_delete_removes_the_checkpoint_file(agent_checkpoint_service: AgentCheckpointService) -> None:
    await agent_checkpoint_service.save("key-1", make_checkpoint("key-1"))
    assert await agent_checkpoint_service.load("key-1") is not None

    await agent_checkpoint_service.delete("key-1")

    assert await agent_checkpoint_service.load("key-1") is None


@pytest.mark.asyncio
async def test_delete_is_noop_when_no_file_exists(agent_checkpoint_service: AgentCheckpointService) -> None:
    await agent_checkpoint_service.delete("never-saved-key")


@pytest.mark.asyncio
async def test_different_keys_map_to_different_files(agent_checkpoint_service: AgentCheckpointService) -> None:
    await agent_checkpoint_service.save("key-1", make_checkpoint("key-1"))
    await agent_checkpoint_service.save("key-2", make_checkpoint("key-2"))

    assert len(list(agent_checkpoint_service.checkpoint_dir.iterdir())) == 2
    assert (await agent_checkpoint_service.load("key-1")).key == "key-1"
    assert (await agent_checkpoint_service.load("key-2")).key == "key-2"
