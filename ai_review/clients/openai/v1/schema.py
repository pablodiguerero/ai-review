from typing import Any, Literal

from pydantic import BaseModel


class OpenAIUsageSchema(BaseModel):
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int


class OpenAIMessageSchema(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OpenAIChoiceSchema(BaseModel):
    message: OpenAIMessageSchema


class OpenAIStreamOptionsSchema(BaseModel):
    include_usage: bool = True


class OpenAIChatRequestSchema(BaseModel):
    model: str
    stream: bool = False
    stream_options: OpenAIStreamOptionsSchema | None = None
    messages: list[OpenAIMessageSchema]
    max_tokens: int | None = None
    temperature: float | None = None
    response_format: dict | None = None


class OpenAIStreamDeltaSchema(BaseModel):
    content: str | None = None
    # Never rendered - tracked only because a provider that streamed reasoning
    # has already billed for it, which decides whether a retry is safe. Both
    # spellings are in the wild.
    reasoning: str | None = None
    reasoning_content: str | None = None

    @property
    def is_billable(self) -> bool:
        return bool(self.content or self.reasoning or self.reasoning_content)


class OpenAIStreamChoiceSchema(BaseModel):
    index: int | None = None
    delta: OpenAIStreamDeltaSchema | None = None
    finish_reason: str | None = None


# Deliberately looser than OpenAIUsageSchema: gateways attach partial or
# differently named usage blocks to content chunks, and a strict model would
# drop the chunk together with its text.
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
    usage: OpenAIUsageSchema
    choices: list[OpenAIChoiceSchema]

    @property
    def first_text(self) -> str:
        if not self.choices:
            return ""

        return (self.choices[0].message.content or "").strip()
