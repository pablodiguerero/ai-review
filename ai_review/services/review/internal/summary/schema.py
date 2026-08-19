from pydantic import BaseModel, field_validator

from ai_review.config import settings


class SummaryCommentSchema(BaseModel):
    text: str

    @field_validator("text")
    def normalize_text(cls, value: str) -> str:
        return (value or "").strip()

    @property
    def body_with_tag(self) -> str:
        header = settings.review.summary_header
        if header:
            rendered_header = header.format(model=settings.llm.meta.model)
            return f"{rendered_header}\n\n{self.text}\n\n{settings.review.summary_tag}"

        return f"{self.text}\n\n{settings.review.summary_tag}"

    @property
    def body_with_fallback_tag(self) -> str:
        return f"{self.text}\n\n{settings.review.inline_fallback_tag}"
