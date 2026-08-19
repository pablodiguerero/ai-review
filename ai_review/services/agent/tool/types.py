from typing import Protocol

from ai_review.services.agent.tool.schema import AgentToolResultSchema


class AgentToolServiceProtocol(Protocol):
    async def execute(self, command: str) -> AgentToolResultSchema:
        ...
