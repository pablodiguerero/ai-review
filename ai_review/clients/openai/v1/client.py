from httpx import Response, AsyncHTTPTransport, AsyncClient, TransportError
from pydantic import ValidationError

from ai_review.clients.openai.v1.schema import (
    OpenAIChatRequestSchema,
    OpenAIChatResponseSchema,
    OpenAIChatStreamChunkSchema,
    OpenAIChoiceSchema,
    OpenAIMessageSchema,
    OpenAIStreamUsageSchema,
    OpenAIUsageSchema,
)
from ai_review.clients.openai.v1.types import OpenAIV1HTTPClientProtocol
from ai_review.config import settings
from ai_review.libs.http.client import HTTPClient
from ai_review.libs.http.event_hooks.logger import LoggerEventHook
from ai_review.libs.http.handlers import HTTPClientError, handle_http_error
from ai_review.libs.http.transports.retry import RetryTransport
from ai_review.libs.logger import get_logger


STREAM_ATTEMPTS = 2

# Gateways differ on how they end a stream: some send [DONE], some only a usage
# frame, some a finish_reason. Empty strings and nulls are not an ending - one
# gateway sends "finish_reason": "" on every content chunk.
TERMINAL_FINISH_REASONS = ("stop", "length", "content_filter", "tool_calls")

logger = get_logger("OPENAI_V1_HTTP_CLIENT")


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

    def stream_error(self, details: str, status_code: int = 502) -> OpenAIV1HTTPClientError:
        return OpenAIV1HTTPClientError(
            client="OpenAIV1HTTPClient", details=details, status_code=status_code
        )

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
                    raise self.stream_error(
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
                        # Dropping a frame that carried text would silently
                        # shorten the answer; anything else is gateway noise.
                        if '"content"' in payload:
                            raise self.stream_error(
                                details=f"Unparsable stream chunk carried content: {error}"
                            ) from error

                        logger.warning(f"Skipping unparsable stream chunk: {error}")
                        continue

                    if chunk.error:
                        raise self.stream_error(details=f"Stream carried an error: {chunk.error}")

                    # Keep the latest non-empty usage block: gateways report a
                    # running total and then an all-zero frame, which would erase it.
                    if chunk.usage is not None:
                        if not chunk.usage.is_empty:
                            usage = chunk.usage

                        # Counted output is work the provider already did.
                        if chunk.usage.completion_tokens:
                            billed = True

                        # ...but only a usage frame that carries no choices is
                        # the gateway's closing frame. The same field rides along
                        # content chunks as a running total, and reading that as
                        # an ending would turn a mid-answer drop into a complete
                        # review. Some models give no other end signal at all.
                        if chunk.usage.completion_tokens and not chunk.choices:
                            saw_usage = True

                    for choice in chunk.choices:
                        if choice.index not in (None, 0):
                            continue

                        if choice.finish_reason:
                            finish_reason = choice.finish_reason

                        if not choice.delta:
                            continue

                        # Reasoning is billed like output even though it is never
                        # rendered, so it counts as work the provider already did.
                        if choice.delta.is_billable:
                            billed = True

                        if choice.delta.content:
                            parts.append(choice.delta.content)
        except TransportError as error:
            # A stream that already signalled its end has delivered everything;
            # an error while closing it must not discard a complete answer.
            if completed or finish_reason in TERMINAL_FINISH_REASONS or saw_usage:
                logger.warning(f"Stream errored after it had completed: {error}")
            else:
                raise StreamInterrupted(error, consumed=billed) from error

        # A stream that ends with none of [DONE], a finish_reason or a usage
        # frame was cut mid-answer: returning what arrived would post a truncated
        # review as if it were the whole one. Models differ on which of the three
        # they send - grok on the zen gateway sends only usage - so any one of
        # them counts. An empty answer is NOT an error here: the agent loop
        # retries those itself.
        if not (completed or finish_reason in TERMINAL_FINISH_REASONS or saw_usage):
            raise self.stream_error(details="Stream ended with no completion signal")

        # max_tokens truncation is not a shorter answer, it is half an answer -
        # posting it as a review is exactly what the [DONE] check exists to stop.
        if finish_reason == "length":
            raise self.stream_error(details="Stream hit the max_tokens limit mid-answer")

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
        # Retry only while nothing has been generated yet. Once the provider has
        # streamed content it has billed for it, and a second attempt would pay
        # for the whole answer twice.
        for attempt in range(1, STREAM_ATTEMPTS + 1):
            try:
                return await self.read_stream(request)
            except StreamInterrupted as interrupted:
                if interrupted.consumed or attempt == STREAM_ATTEMPTS:
                    raise self.stream_error(
                        details=f"Stream transport error: {interrupted.error}"
                    ) from interrupted.error

                logger.warning(
                    f"Stream interrupted before any content, retrying: {interrupted.error}"
                )

        raise self.stream_error(details="Stream was never opened")

    async def chat(self, request: OpenAIChatRequestSchema) -> OpenAIChatResponseSchema:
        if request.stream:
            return await self.chat_stream(request)

        response = await self.chat_api(request)
        return OpenAIChatResponseSchema.model_validate_json(response.text)


def get_openai_v1_http_client() -> OpenAIV1HTTPClient:
    logger = get_logger("OPENAI_V1_HTTP_CLIENT")
    logger_event_hook = LoggerEventHook(logger=logger)
    retry_transport = RetryTransport(
        logger=logger,
        transport=AsyncHTTPTransport(
            proxy=settings.llm.http_client.proxy_url_value,
            verify=settings.llm.http_client.verify
        )
    )

    client = AsyncClient(
        verify=settings.llm.http_client.verify,
        timeout=settings.llm.http_client.timeout,
        headers={"Authorization": f"Bearer {settings.llm.http_client.api_token_value}"},
        base_url=settings.llm.http_client.api_url_value,
        transport=retry_transport,
        event_hooks={
            'request': [logger_event_hook.request],
            'response': [logger_event_hook.response]
        }
    )

    return OpenAIV1HTTPClient(client=client)
