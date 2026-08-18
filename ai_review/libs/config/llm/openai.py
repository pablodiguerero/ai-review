from pydantic import model_validator

from ai_review.libs.config.http import HTTPClientWithTokenConfig
from ai_review.libs.config.llm.meta import LLMMetaConfig


class OpenAIMetaConfig(LLMMetaConfig):
    model: str = "gpt-4o-mini"
    stream: bool = False

    @property
    def is_v2_model(self) -> bool:
        return any(self.model.startswith(model) for model in ("gpt-5", "gpt-4.1"))

    @model_validator(mode="after")
    def validate_stream_is_supported(self):
        # Fail loudly instead of silently sending a non-streaming request: this
        # client has no streaming branch for the responses API.
        if self.stream and self.is_v2_model:
            raise ValueError(f"Streaming is not implemented for the responses API model {self.model}")

        return self


class OpenAIHTTPClientConfig(HTTPClientWithTokenConfig):
    pass
