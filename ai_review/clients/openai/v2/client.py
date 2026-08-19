from httpx import Response, AsyncClient, AsyncHTTPTransport, TransportError, Timeout
from pydantic import ValidationError

from ai_review.clients.openai.v2.schema import (
    OpenAIResponseContentSchema,
    OpenAIResponseOutputSchema,
    OpenAIResponseUsageSchema,
    OpenAIResponsesRequestSchema,
    OpenAIResponsesResponseSchema,
    OpenAIResponsesStreamEventSchema,
)
from ai_review.clients.openai.v2.types import OpenAIV2HTTPClientProtocol
from ai_review.clients.openai.streaming import StreamInterrupted
from ai_review.config import settings
from ai_review.libs.http.client import HTTPClient
from ai_review.libs.http.event_hooks.logger import LoggerEventHook
from ai_review.libs.http.handlers import HTTPClientError, handle_http_error
from ai_review.libs.http.transports.retry import RetryTransport
from ai_review.libs.logger import get_logger


STREAM_ATTEMPTS = 2

MAX_OUTPUT_TOKENS_ERROR_DETAILS = "Response hit the max_output_tokens limit mid-answer"

logger = get_logger("OPENAI_V2_HTTP_CLIENT")


def is_reasoning_delta_event(event_type: str) -> bool:
    return "reasoning" in event_type and "delta" in event_type


def usage_from_response_payload(response_payload: dict) -> OpenAIResponseUsageSchema:
    return OpenAIResponseUsageSchema.model_validate(response_payload.get("usage") or {})


def incomplete_reason_from_response_payload(response_payload: dict) -> str | None:
    return (response_payload.get("incomplete_details") or {}).get("reason")


def error_message_from_event(event: OpenAIResponsesStreamEventSchema, payload: str) -> str:
    if event.message:
        return event.message

    error_payload = (event.response or {}).get("error") or {}
    if isinstance(error_payload, dict) and error_payload.get("message"):
        return error_payload["message"]

    return payload


class OpenAIV2HTTPClientError(HTTPClientError):
    pass


class OpenAIV2HTTPClient(HTTPClient, OpenAIV2HTTPClientProtocol):
    @handle_http_error(client='OpenAIV2HTTPClient', exception=OpenAIV2HTTPClientError)
    async def chat_api(self, request: OpenAIResponsesRequestSchema) -> Response:
        return await self.post("/responses", json=request.model_dump(exclude_none=True))

    def build_error(self, details: str, status_code: int = 502) -> OpenAIV2HTTPClientError:
        return OpenAIV2HTTPClientError(
            client="OpenAIV2HTTPClient", details=details, status_code=status_code
        )

    async def read_stream(self, request: OpenAIResponsesRequestSchema) -> OpenAIResponsesResponseSchema:
        parts: list[str] = []
        usage: OpenAIResponseUsageSchema | None = None
        billed = False
        completed = False

        try:
            async with self.client.stream(
                    "POST", "/responses", json=request.model_dump(exclude_none=True)
            ) as response:
                if response.is_error:
                    await response.aread()
                    raise self.build_error(
                        details=response.text or "OpenAIV2HTTPClient returned error",
                        status_code=response.status_code,
                    )

                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue

                    payload = line[len("data:"):].strip()
                    if not payload:
                        continue

                    try:
                        event = OpenAIResponsesStreamEventSchema.model_validate_json(payload)
                    except ValidationError as error:
                        logger.warning(f"Skipping unparsable stream event: {error}")
                        continue

                    if event.type == "response.output_text.delta":
                        billed = True
                        if event.delta:
                            parts.append(event.delta)
                        continue

                    if is_reasoning_delta_event(event.type):
                        billed = True
                        continue

                    if event.type in ("response.failed", "error"):
                        raise self.build_error(
                            details=f"Stream carried an error: {error_message_from_event(event, payload)}"
                        )

                    if event.type in ("response.completed", "response.incomplete"):
                        response_payload = event.response or {}

                        if event.type == "response.incomplete":
                            if incomplete_reason_from_response_payload(response_payload) == "max_output_tokens":
                                raise self.build_error(details=MAX_OUTPUT_TOKENS_ERROR_DETAILS)

                        usage = usage_from_response_payload(response_payload)
                        completed = True
        except TransportError as error:
            if completed:
                logger.warning(f"Stream errored after it had completed: {error}")
            else:
                raise StreamInterrupted(error, consumed=billed) from error

        if not completed:
            raise self.build_error(details="Stream ended with no completion signal")

        return OpenAIResponsesResponseSchema(
            usage=usage or OpenAIResponseUsageSchema(),
            output=[
                OpenAIResponseOutputSchema(
                    type="message",
                    role="assistant",
                    content=[OpenAIResponseContentSchema(type="output_text", text="".join(parts))],
                )
            ],
            status="completed",
        )

    async def chat_stream(self, request: OpenAIResponsesRequestSchema) -> OpenAIResponsesResponseSchema:
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

    async def chat(self, request: OpenAIResponsesRequestSchema) -> OpenAIResponsesResponseSchema:
        if request.stream:
            return await self.chat_stream(request)

        response = await self.chat_api(request)
        chat_response = OpenAIResponsesResponseSchema.model_validate_json(response.text)

        incomplete_reason = (chat_response.incomplete_details or {}).get("reason")
        if chat_response.status == "incomplete" and incomplete_reason == "max_output_tokens":
            raise self.build_error(details=MAX_OUTPUT_TOKENS_ERROR_DETAILS)

        return chat_response


def get_openai_v2_http_client() -> OpenAIV2HTTPClient:
    logger = get_logger("OPENAI_V2_HTTP_CLIENT")
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

    return OpenAIV2HTTPClient(client=client)
