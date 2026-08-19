from pathlib import Path
from typing import Protocol

from ai_review.services.artifacts.schema.base import BaseArtifactSchema
from ai_review.services.artifacts.schema.llm import LLMArtifactAgentSchema
from ai_review.services.cost.schema import CostReportSchema
from ai_review.services.review.internal.inline.schema import InlineCommentSchema
from ai_review.services.review.internal.inline_reply.schema import InlineCommentReplySchema
from ai_review.services.review.internal.summary.schema import SummaryCommentSchema
from ai_review.services.review.internal.summary_reply.schema import SummaryCommentReplySchema


class ArtifactsServiceProtocol(Protocol):
    async def save(
            self,
            artifact: BaseArtifactSchema,
            artifacts_dir: Path,
            artifacts_enabled: bool,
    ) -> str | None:
        ...

    async def save_llm(
            self,
            prompt: str,
            response: str,
            prompt_system: str,
            cost_report: CostReportSchema | None = None,
            agent: LLMArtifactAgentSchema | None = None,
    ) -> str | None:
        ...

    async def save_vcs_inline(self, comment: InlineCommentSchema) -> str | None:
        ...

    async def save_vcs_summary(self, comment: SummaryCommentSchema) -> str | None:
        ...

    async def save_vcs_inline_reply(self, thread_id: str, reply: InlineCommentReplySchema) -> str | None:
        ...

    async def save_vcs_summary_reply(self, thread_id: str, reply: SummaryCommentReplySchema) -> str | None:
        ...
