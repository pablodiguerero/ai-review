from httpx import Response, AsyncClient, AsyncHTTPTransport, Timeout

from ai_review.clients.openai.v2.schema import (
    OpenAIResponsesRequestSchema,
    OpenAIResponsesResponseSchema
)
from ai_review.clients.openai.v2.types import OpenAIV2HTTPClientProtocol
from ai_review.config import settings
from ai_review.libs.http.client import HTTPClient
from ai_review.libs.http.event_hooks.logger import LoggerEventHook
from ai_review.libs.http.handlers import HTTPClientError, handle_http_error
from ai_review.libs.http.transports.retry import RetryTransport
from ai_review.libs.logger import get_logger


class OpenAIV2HTTPClientError(HTTPClientError):
    pass


class OpenAIV2HTTPClient(HTTPClient, OpenAIV2HTTPClientProtocol):
    @handle_http_error(client='OpenAIV2HTTPClient', exception=OpenAIV2HTTPClientError)
    async def chat_api(self, request: OpenAIResponsesRequestSchema) -> Response:
        return await self.post("/responses", json=request.model_dump(exclude_none=True))

    async def chat(self, request: OpenAIResponsesRequestSchema) -> OpenAIResponsesResponseSchema:
        response = await self.chat_api(request)
        chat_response = OpenAIResponsesResponseSchema.model_validate_json(response.text)

        incomplete_reason = (chat_response.incomplete_details or {}).get("reason")
        if chat_response.status == "incomplete" and incomplete_reason == "max_output_tokens":
            raise OpenAIV2HTTPClientError(
                client="OpenAIV2HTTPClient",
                details="Response hit the max_output_tokens limit mid-answer",
                status_code=502,
            )

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
