from enum import StrEnum
from typing import Any

from pydantic import model_validator

from ai_review.libs.config.http import HTTPClientWithTokenConfig
from ai_review.libs.config.llm.meta import LLMMetaConfig


class OpenAIAPI(StrEnum):
    AUTO = "AUTO"
    CHAT = "CHAT"
    RESPONSES = "RESPONSES"


RESERVED_EXTRA_BODY_KEYS = ("stream", "stream_options", "messages", "input", "model", "response_format", "text")


class OpenAIMetaConfig(LLMMetaConfig):
    model: str = "gpt-4o-mini"
    stream: bool = False
    api: OpenAIAPI = OpenAIAPI.AUTO
    extra_body: dict[str, Any] | None = None

    @property
    def is_v2_model(self) -> bool:
        return any(self.model.startswith(model) for model in ("gpt-5", "gpt-4.1"))

    @property
    def use_responses_api(self) -> bool:
        if self.api is OpenAIAPI.CHAT:
            return False

        if self.api is OpenAIAPI.RESPONSES:
            return True

        return self.is_v2_model

    @model_validator(mode="after")
    def validate_stream_is_supported(self):
        if self.stream and self.use_responses_api:
            raise ValueError(f"Streaming is not implemented for the responses API model {self.model}")

        return self

    @model_validator(mode="after")
    def validate_extra_body_does_not_override_reserved_keys(self):
        if not self.extra_body:
            return self

        for key in RESERVED_EXTRA_BODY_KEYS:
            if key in self.extra_body:
                raise ValueError(f"extra_body must not override reserved request field {key!r}")

        return self


class OpenAIHTTPClientConfig(HTTPClientWithTokenConfig):
    pass
