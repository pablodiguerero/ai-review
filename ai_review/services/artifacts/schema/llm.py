from pydantic import BaseModel

from ai_review.services.artifacts.schema.base import BaseArtifactSchema, ArtifactType
from ai_review.services.cost.schema import CostReportSchema


class LLMArtifactAgentSchema(BaseModel):
    iterations: int
    executed_tool_calls: int
    blocked_tool_calls: int
    stop_reason: str


class LLMArtifactDataSchema(BaseModel):
    prompt: str
    response: str
    cost_report: CostReportSchema | None = None
    prompt_system: str
    agent: LLMArtifactAgentSchema | None = None


class LLMArtifactSchema(BaseArtifactSchema[LLMArtifactDataSchema]):
    type: ArtifactType = ArtifactType.LLM
