from ai_review.config import settings
from ai_review.libs.logger import get_logger
from ai_review.libs.markdown.tables import normalize_markdown_tables
from ai_review.services.review.internal.summary.schema import SummaryCommentSchema
from ai_review.services.review.internal.summary.types import SummaryCommentServiceProtocol

logger = get_logger("SUMMARY_COMMENT_SERVICE")


class SummaryCommentService(SummaryCommentServiceProtocol):
    @classmethod
    def parse_model_output(cls, output: str) -> SummaryCommentSchema:
        text = (output or "").strip()
        if not text:
            logger.warning("LLM returned empty summary")

        if text and settings.review.summary_normalize_tables:
            text = normalize_markdown_tables(text)

        return SummaryCommentSchema(text=text)
