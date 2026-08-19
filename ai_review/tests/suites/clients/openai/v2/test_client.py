import pytest
from httpx import AsyncClient, MockTransport, Response

from ai_review.clients.openai.v2.client import (
    get_openai_v2_http_client,
    OpenAIV2HTTPClient,
    OpenAIV2HTTPClientError,
)
from ai_review.clients.openai.v2.schema import OpenAIInputMessageSchema, OpenAIResponsesRequestSchema


@pytest.mark.usefixtures('openai_v2_http_client_config')
def test_get_openai_v2_http_client_builds_ok():
    openai_http_client = get_openai_v2_http_client()

    assert isinstance(openai_http_client, OpenAIV2HTTPClient)
    assert isinstance(openai_http_client.client, AsyncClient)

    timeout = openai_http_client.client.timeout
    assert timeout.connect == 3
    assert timeout.read == 10
    assert openai_http_client.client._transport.retry_transport_errors is True


def build_v2_client(handler) -> OpenAIV2HTTPClient:
    return OpenAIV2HTTPClient(
        client=AsyncClient(base_url="https://api.openai.com/v1", transport=MockTransport(handler))
    )


def build_responses_request(**kwargs) -> OpenAIResponsesRequestSchema:
    return OpenAIResponsesRequestSchema(
        model="gpt-5",
        input=[OpenAIInputMessageSchema(role="user", content="hi")],
        **kwargs
    )


@pytest.mark.asyncio
async def test_chat_returns_response_when_complete():
    body = {
        "usage": {"total_tokens": 10, "input_tokens": 4, "output_tokens": 6},
        "output": [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hello"}]}
        ],
        "status": "completed",
    }
    client = build_v2_client(lambda request: Response(200, json=body))

    response = await client.chat(build_responses_request())

    assert response.first_text == "hello"


@pytest.mark.asyncio
async def test_chat_raises_when_incomplete_due_to_max_output_tokens():
    body = {
        "usage": {"total_tokens": 10, "input_tokens": 4, "output_tokens": 6},
        "output": [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "half"}]}
        ],
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
    }
    client = build_v2_client(lambda request: Response(200, json=body))

    with pytest.raises(OpenAIV2HTTPClientError, match="max_output_tokens") as error:
        await client.chat(build_responses_request())

    assert error.value.status_code == 502


@pytest.mark.asyncio
async def test_chat_does_not_raise_when_incomplete_for_another_reason():
    body = {
        "usage": {"total_tokens": 10, "input_tokens": 4, "output_tokens": 6},
        "output": [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "half"}]}
        ],
        "status": "incomplete",
        "incomplete_details": {"reason": "content_filter"},
    }
    client = build_v2_client(lambda request: Response(200, json=body))

    response = await client.chat(build_responses_request())

    assert response.first_text == "half"


@pytest.mark.asyncio
async def test_chat_does_not_raise_when_incomplete_details_missing():
    body = {
        "usage": {"total_tokens": 10, "input_tokens": 4, "output_tokens": 6},
        "output": [],
        "status": "incomplete",
    }
    client = build_v2_client(lambda request: Response(200, json=body))

    response = await client.chat(build_responses_request())

    assert response.first_text == ""
