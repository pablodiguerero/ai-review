from pathlib import Path

import pytest

from ai_review.services.agent.checkpoint.schema import AgentCheckpointSchema
from ai_review.services.agent.checkpoint.service import AgentCheckpointService
from ai_review.services.agent.checkpoint.types import AgentCheckpointServiceProtocol


class FakeAgentCheckpointService(AgentCheckpointServiceProtocol):
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.store: dict[str, AgentCheckpointSchema] = {}

    async def load(self, key: str) -> AgentCheckpointSchema | None:
        self.calls.append(("load", {"key": key}))
        return self.store.get(key)

    async def save(self, key: str, checkpoint: AgentCheckpointSchema) -> None:
        self.calls.append(("save", {"key": key, "checkpoint": checkpoint}))
        self.store[key] = checkpoint

    async def delete(self, key: str) -> None:
        self.calls.append(("delete", {"key": key}))
        self.store.pop(key, None)


@pytest.fixture
def fake_agent_checkpoint_service() -> FakeAgentCheckpointService:
    return FakeAgentCheckpointService()


@pytest.fixture
def agent_checkpoint_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AgentCheckpointService:
    checkpoint_dir = tmp_path / "checkpoints"
    monkeypatch.setattr("ai_review.config.settings.agent.checkpoint_dir", checkpoint_dir)
    return AgentCheckpointService()
