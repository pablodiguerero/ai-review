from ai_review.clients.openai.v1.client import get_openai_v1_http_client
from ai_review.clients.openai.v1.schema import (
    OpenAIChatRequestSchema,
    OpenAIMessageSchema,
    OpenAIStreamOptionsSchema,
)
from ai_review.clients.openai.v2.client import get_openai_v2_http_client
from ai_review.clients.openai.v2.schema import OpenAIInputMessageSchema, OpenAIResponsesRequestSchema
from ai_review.config import settings
from ai_review.services.llm.types import LLMClientProtocol, ChatResultSchema


class OpenAILLMClient(LLMClientProtocol):
    def __init__(self):
        self.meta = settings.llm.meta

        self.http_client_v1 = get_openai_v1_http_client()
        self.http_client_v2 = get_openai_v2_http_client()

    async def chat_v1(self, prompt: str, prompt_system: str, json_mode: bool = False) -> ChatResultSchema:
        request_fields = {
            "model": self.meta.model,
            "messages": [
                OpenAIMessageSchema(role="system", content=prompt_system),
                OpenAIMessageSchema(role="user", content=prompt),
            ],
            "max_tokens": self.meta.max_tokens,
            "temperature": self.meta.temperature,
            "response_format": {"type": "json_object"} if json_mode else None,
            "stream": self.meta.stream,
            "stream_options": OpenAIStreamOptionsSchema() if self.meta.stream else None,
        }
        request_fields.update(self.meta.extra_body or {})
        request = OpenAIChatRequestSchema(**request_fields)
        response = await self.http_client_v1.chat(request)
        return ChatResultSchema(
            text=response.first_text,
            total_tokens=response.usage.total_tokens,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

    async def chat_v2(self, prompt: str, prompt_system: str, json_mode: bool = False) -> ChatResultSchema:
        request_fields = {
            "model": self.meta.model,
            "input": [
                OpenAIInputMessageSchema(role="system", content=prompt_system),
                OpenAIInputMessageSchema(role="user", content=prompt),
            ],
            "temperature": self.meta.temperature,
            "max_output_tokens": self.meta.max_tokens,
            "text": {"format": {"type": "json_object"}} if json_mode else None,
        }
        request_fields.update(self.meta.extra_body or {})
        request = OpenAIResponsesRequestSchema(**request_fields)
        response = await self.http_client_v2.chat(request)
        return ChatResultSchema(
            text=response.first_text,
            total_tokens=response.usage.total_tokens,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
        )

    async def chat(self, prompt: str, prompt_system: str, json_mode: bool = False) -> ChatResultSchema:
        if self.meta.use_responses_api:
            return await self.chat_v2(prompt, prompt_system, json_mode=json_mode)

        return await self.chat_v1(prompt, prompt_system, json_mode=json_mode)
