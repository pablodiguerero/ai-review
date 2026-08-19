from typing import Protocol

from ai_review.services.agent.checkpoint.schema import AgentCheckpointSchema


class AgentCheckpointServiceProtocol(Protocol):
    async def load(self, key: str) -> AgentCheckpointSchema | None:
        ...

    async def save(self, key: str, checkpoint: AgentCheckpointSchema) -> None:
        ...

    async def delete(self, key: str) -> None:
        ...
