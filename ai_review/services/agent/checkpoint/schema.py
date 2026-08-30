from enum import StrEnum

from pydantic import BaseModel, Field

from ai_review.services.agent.loop.schema import AgentTraceSchema


class AgentCheckpointStage(StrEnum):
    INVESTIGATING = "investigating"
    NEEDS_FINAL = "needs_final"
    REVIEWED = "reviewed"


class AgentCheckpointSchema(BaseModel):
    key: str
    head_sha: str = ""
    round: int = 0
    stage: AgentCheckpointStage
    traces: list[AgentTraceSchema] = Field(default_factory=list)
    signatures: list[str] = Field(default_factory=list)
    executed_tool_calls: int = 0
    blocked_tool_calls: int = 0
    iterations: int = 0
    context_used: int = 0
    final_replays: int = 0
    prior_synopsis: str = ""
    created_at: str
    updated_at: str
