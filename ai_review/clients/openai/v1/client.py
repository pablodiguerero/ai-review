from httpx import Response, AsyncHTTPTransport, AsyncClient, TransportError, Timeout
from pydantic import ValidationError

from ai_review.clients.openai.v1.schema import (
    OpenAIChatRequestSchema,
    OpenAIChatResponseSchema,
    OpenAIChatStreamChunkSchema,
    OpenAIChoiceSchema,
    OpenAIMessageSchema,
    OpenAIStreamChoiceSchema,
    OpenAIStreamUsageSchema,
)
from ai_review.clients.openai.v1.types import OpenAIV1HTTPClientProtocol
from ai_review.config import settings
from ai_review.libs.http.client import HTTPClient
from ai_review.libs.http.event_hooks.logger import LoggerEventHook
from ai_review.libs.http.handlers import HTTPClientError, handle_http_error
from ai_review.libs.http.transports.retry import RetryTransport
from ai_review.libs.logger import get_logger


STREAM_ATTEMPTS = 2

TERMINAL_FINISH_REASONS = ("stop", "length", "content_filter", "tool_calls")

logger = get_logger("OPENAI_V1_HTTP_CLIENT")


def stream_reached_a_completion_signal(completed: bool, finish_reason: str | None, saw_usage: bool) -> bool:
    return completed or finish_reason in TERMINAL_FINISH_REASONS or saw_usage


def chunk_payload_carries_content(payload: str) -> bool:
    return '"content"' in payload


class OpenAIV1HTTPClientError(HTTPClientError):
    pass


class StreamInterrupted(Exception):
    def __init__(self, error: Exception, consumed: bool):
        self.error = error
        self.consumed = consumed

        super().__init__(str(error))


class OpenAIV1HTTPClient(HTTPClient, OpenAIV1HTTPClientProtocol):
    @handle_http_error(client='OpenAIV1HTTPClient', exception=OpenAIV1HTTPClientError)
    async def chat_api(self, request: OpenAIChatRequestSchema) -> Response:
        return await self.post("/chat/completions", json=request.model_dump(exclude_none=True))

    def build_error(self, details: str, status_code: int = 502) -> OpenAIV1HTTPClientError:
        return OpenAIV1HTTPClientError(
            client="OpenAIV1HTTPClient", details=details, status_code=status_code
        )

    def apply_stream_usage(
            self,
            chunk_usage: OpenAIStreamUsageSchema | None,
            chunk_has_choices: bool,
            usage: OpenAIStreamUsageSchema | None,
            billed: bool,
            saw_usage: bool,
    ) -> tuple[OpenAIStreamUsageSchema | None, bool, bool]:
        if chunk_usage is None:
            return usage, billed, saw_usage

        if not chunk_usage.is_empty:
            usage = chunk_usage

        if chunk_usage.completion_tokens:
            billed = True

        is_closing_usage_frame = bool(chunk_usage.completion_tokens) and not chunk_has_choices
        if is_closing_usage_frame:
            saw_usage = True

        return usage, billed, saw_usage

    def apply_stream_choice(
            self,
            choice: OpenAIStreamChoiceSchema,
            parts: list[str],
            billed: bool,
            finish_reason: str | None,
    ) -> tuple[bool, str | None]:
        if choice.index not in (None, 0):
            return billed, finish_reason

        if choice.finish_reason:
            finish_reason = choice.finish_reason

        if not choice.delta:
            return billed, finish_reason

        if choice.delta.is_billable:
            billed = True

        if choice.delta.content:
            parts.append(choice.delta.content)

        return billed, finish_reason

    async def read_stream(self, request: OpenAIChatRequestSchema) -> OpenAIChatResponseSchema:
        parts: list[str] = []
        usage: OpenAIStreamUsageSchema | None = None
        billed = False
        completed = False
        saw_usage = False
        finish_reason: str | None = None

        try:
            async with self.client.stream(
                    "POST", "/chat/completions", json=request.model_dump(exclude_none=True)
            ) as response:
                if response.is_error:
                    await response.aread()
                    raise self.build_error(
                        details=response.text or "OpenAIV1HTTPClient returned error",
                        status_code=response.status_code,
                    )

                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue

                    payload = line[len("data:"):].strip()
                    if not payload:
                        continue

                    if payload == "[DONE]":
                        completed = True
                        break

                    try:
                        chunk = OpenAIChatStreamChunkSchema.model_validate_json(payload)
                    except ValidationError as error:
                        if chunk_payload_carries_content(payload):
                            raise self.build_error(
                                details=f"Unparsable stream chunk carried content: {error}"
                            ) from error

                        logger.warning(f"Skipping unparsable stream chunk: {error}")
                        continue

                    if chunk.error:
                        raise self.build_error(details=f"Stream carried an error: {chunk.error}")

                    usage, billed, saw_usage = self.apply_stream_usage(
                        chunk.usage, bool(chunk.choices), usage, billed, saw_usage
                    )

                    for choice in chunk.choices:
                        billed, finish_reason = self.apply_stream_choice(choice, parts, billed, finish_reason)
        except TransportError as error:
            if stream_reached_a_completion_signal(completed, finish_reason, saw_usage):
                logger.warning(f"Stream errored after it had completed: {error}")
            else:
                raise StreamInterrupted(error, consumed=billed) from error

        if not stream_reached_a_completion_signal(completed, finish_reason, saw_usage):
            raise self.build_error(details="Stream ended with no completion signal")

        if finish_reason == "length":
            raise self.build_error(details="Stream hit the max_tokens limit mid-answer")

        if usage is None:
            logger.warning("Stream carried no usage block, reporting zero tokens")

        return OpenAIChatResponseSchema(
            usage=(usage or OpenAIStreamUsageSchema()).to_usage(),
            choices=[
                OpenAIChoiceSchema(
                    message=OpenAIMessageSchema(role="assistant", content="".join(parts))
                )
            ],
        )

    async def chat_stream(self, request: OpenAIChatRequestSchema) -> OpenAIChatResponseSchema:
        for attempt in range(1, STREAM_ATTEMPTS + 1):
            try:
                return await self.read_stream(request)
            except StreamInterrupted as interrupted:
                if interrupted.consumed or attempt == STREAM_ATTEMPTS:
                    raise self.build_error(
                        details=f"Stream transport error: {interrupted.error}"
                    ) from interrupted.error

                logger.warning(
                    f"Stream interrupted before any content, retrying: {interrupted.error}"
                )

        raise self.build_error(details="Stream was never opened")

    async def chat(self, request: OpenAIChatRequestSchema) -> OpenAIChatResponseSchema:
        if request.stream:
            return await self.chat_stream(request)

        response = await self.chat_api(request)
        chat_response = OpenAIChatResponseSchema.model_validate_json(response.text)

        if chat_response.choices and chat_response.choices[0].finish_reason == "length":
            raise self.build_error(details="Response hit the max_tokens limit mid-answer")

        return chat_response


def get_openai_v1_http_client() -> OpenAIV1HTTPClient:
    logger = get_logger("OPENAI_V1_HTTP_CLIENT")
    logger_event_hook = LoggerEventHook(logger=logger)
    retry_transport = RetryTransport(
        logger=logger,
        transport=AsyncHTTPTransport(
            proxy=settings.llm.http_client.proxy_url_value,
            verify=settings.llm.http_client.verify
        ),
        retry_transport_errors=True,
    )

    client = AsyncClient(
        verify=settings.llm.http_client.verify,
        timeout=Timeout(
            connect=settings.llm.http_client.connect_timeout,
            read=settings.llm.http_client.timeout,
            write=settings.llm.http_client.timeout,
            pool=settings.llm.http_client.timeout,
        ),
        headers={"Authorization": f"Bearer {settings.llm.http_client.api_token_value}"},
        base_url=settings.llm.http_client.api_url_value,
        transport=retry_transport,
        event_hooks={
            'request': [logger_event_hook.request],
            'response': [logger_event_hook.response]
        }
    )

    return OpenAIV1HTTPClient(client=client)
