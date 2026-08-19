from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class OpenAIUsageSchema(BaseModel):
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class OpenAIMessageSchema(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OpenAIChoiceSchema(BaseModel):
    message: OpenAIMessageSchema
    finish_reason: str | None = None


class OpenAIStreamOptionsSchema(BaseModel):
    include_usage: bool = True


class OpenAIChatRequestSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    stream: bool = False
    stream_options: OpenAIStreamOptionsSchema | None = None
    messages: list[OpenAIMessageSchema]
    max_tokens: int | None = None
    temperature: float | None = None
    response_format: dict | None = None


class OpenAIStreamDeltaSchema(BaseModel):
    content: str | None = None
    reasoning: str | None = None
    reasoning_content: str | None = None

    @property
    def is_billable(self) -> bool:
        return bool(self.content or self.reasoning or self.reasoning_content)


class OpenAIStreamChoiceSchema(BaseModel):
    index: int | None = None
    delta: OpenAIStreamDeltaSchema | None = None
    finish_reason: str | None = None


class OpenAIStreamUsageSchema(BaseModel):
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def is_empty(self) -> bool:
        return not (self.total_tokens or self.prompt_tokens or self.completion_tokens)

    def to_usage(self) -> OpenAIUsageSchema:
        return OpenAIUsageSchema(
            total_tokens=self.total_tokens or (self.prompt_tokens + self.completion_tokens),
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
        )


class OpenAIChatStreamChunkSchema(BaseModel):
    error: Any | None = None
    usage: OpenAIStreamUsageSchema | None = None
    choices: list[OpenAIStreamChoiceSchema] = []


class OpenAIChatResponseSchema(BaseModel):
    usage: OpenAIUsageSchema = Field(default_factory=OpenAIUsageSchema)
    choices: list[OpenAIChoiceSchema]

    @property
    def first_text(self) -> str:
        if not self.choices:
            return ""

        return (self.choices[0].message.content or "").strip()
