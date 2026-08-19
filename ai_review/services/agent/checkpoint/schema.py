from pydantic import BaseModel, Field

from ai_review.services.agent.loop.schema import AgentTraceSchema


class AgentCheckpointSchema(BaseModel):
    key: str
    traces: list[AgentTraceSchema] = Field(default_factory=list)
    executed_tool_calls: int = 0
    blocked_tool_calls: int = 0
    iterations: int = 0
    context_used: int = 0
    signatures: list[str] = Field(default_factory=list)
    finished_iterations: bool = False
    created_at: str
