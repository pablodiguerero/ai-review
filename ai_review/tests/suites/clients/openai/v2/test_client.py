import json

import pytest
from httpx import AsyncByteStream, AsyncClient, ConnectError, MockTransport, ReadError, Response

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


class FailingStream(AsyncByteStream):
    def __init__(self, prefix: bytes):
        self.prefix = prefix

    async def __aiter__(self):
        yield self.prefix
        raise ReadError("connection dropped mid-body")


SSE_BODY = (
    b'event: response.created\n'
    b'data: {"type":"response.created","response":{"status":"in_progress"}}\n\n'
    b'event: response.output_text.delta\n'
    b'data: {"type":"response.output_text.delta","delta":"Hello"}\n\n'
    b'event: response.output_text.delta\n'
    b'data: {"type":"response.output_text.delta","delta":" world"}\n\n'
    b'event: response.completed\n'
    b'data: {"type":"response.completed","response":{"status":"completed",'
    b'"usage":{"input_tokens":6,"output_tokens":38,"total_tokens":44},"output":[]}}\n\n'
)


@pytest.mark.asyncio
async def test_chat_stream_collects_deltas_and_usage():
    client = build_v2_client(lambda request: Response(200, content=SSE_BODY))

    response = await client.chat(build_responses_request(stream=True))

    assert response.first_text == "Hello world"
    assert response.usage.input_tokens == 6
    assert response.usage.output_tokens == 38
    assert response.usage.total_tokens == 44
    assert response.status == "completed"


@pytest.mark.asyncio
async def test_chat_stream_sends_stream_flag():
    captured: list[dict] = []

    def handler(request):
        captured.append(json.loads(request.content))
        return Response(200, content=SSE_BODY)

    client = build_v2_client(handler)

    await client.chat(build_responses_request(stream=True))

    assert captured[0]["stream"] is True


@pytest.mark.asyncio
async def test_chat_stream_ignores_reasoning_deltas_in_text():
    body = (
        b'data: {"type":"response.reasoning_summary_text.delta","delta":"thinking"}\n\n'
        b'data: {"type":"response.output_text.delta","delta":"Hello"}\n\n'
        b'data: {"type":"response.completed","response":{"status":"completed","output":[]}}\n\n'
    )
    client = build_v2_client(lambda request: Response(200, content=body))

    response = await client.chat(build_responses_request(stream=True))

    assert response.first_text == "Hello"


@pytest.mark.asyncio
async def test_chat_stream_completed_without_deltas_returns_empty_text():
    body = b'data: {"type":"response.completed","response":{"status":"completed","output":[]}}\n\n'
    client = build_v2_client(lambda request: Response(200, content=body))

    response = await client.chat(build_responses_request(stream=True))

    assert response.first_text == ""
    assert response.usage.total_tokens == 0


@pytest.mark.asyncio
async def test_chat_stream_raises_when_incomplete_due_to_max_output_tokens():
    body = (
        b'data: {"type":"response.output_text.delta","delta":"half"}\n\n'
        b'data: {"type":"response.incomplete","response":{"status":"incomplete",'
        b'"incomplete_details":{"reason":"max_output_tokens"},"output":[]}}\n\n'
    )
    client = build_v2_client(lambda request: Response(200, content=body))

    with pytest.raises(OpenAIV2HTTPClientError, match="max_output_tokens") as error:
        await client.chat(build_responses_request(stream=True))

    assert error.value.status_code == 502


@pytest.mark.asyncio
async def test_chat_stream_does_not_raise_when_incomplete_for_another_reason():
    body = (
        b'data: {"type":"response.output_text.delta","delta":"half"}\n\n'
        b'data: {"type":"response.incomplete","response":{"status":"incomplete",'
        b'"incomplete_details":{"reason":"content_filter"},"output":[]}}\n\n'
    )
    client = build_v2_client(lambda request: Response(200, content=body))

    response = await client.chat(build_responses_request(stream=True))

    assert response.first_text == "half"


@pytest.mark.asyncio
async def test_chat_stream_raises_on_error_event():
    body = b'data: {"type":"error","message":"Upstream request failed"}\n\n'
    client = build_v2_client(lambda request: Response(200, content=body))

    with pytest.raises(OpenAIV2HTTPClientError, match="Upstream request failed"):
        await client.chat(build_responses_request(stream=True))


@pytest.mark.asyncio
async def test_chat_stream_raises_on_response_failed_event():
    body = (
        b'data: {"type":"response.failed","response":{"status":"failed",'
        b'"error":{"message":"gateway exploded"},"output":[]}}\n\n'
    )
    client = build_v2_client(lambda request: Response(200, content=body))

    with pytest.raises(OpenAIV2HTTPClientError, match="gateway exploded"):
        await client.chat(build_responses_request(stream=True))


@pytest.mark.asyncio
async def test_chat_stream_raises_on_error_status():
    client = build_v2_client(lambda request: Response(402, content=b'{"error":"no credits"}'))

    with pytest.raises(OpenAIV2HTTPClientError) as error:
        await client.chat(build_responses_request(stream=True))

    assert error.value.status_code == 402
    assert "no credits" in error.value.details


@pytest.mark.asyncio
async def test_chat_stream_raises_on_no_completion_signal():
    body = b'data: {"type":"response.output_text.delta","delta":"half an ans"}\n\n'
    client = build_v2_client(lambda request: Response(200, content=body))

    with pytest.raises(OpenAIV2HTTPClientError, match="completion signal"):
        await client.chat(build_responses_request(stream=True))


@pytest.mark.asyncio
async def test_chat_stream_skips_malformed_events():
    body = (
        b'data: not-json\n\n'
        b'data: {"type":"response.output_text.delta","delta":"kept"}\n\n'
        b': keep-alive comment\n\n'
        b'data: {"type":"response.completed","response":{"status":"completed","output":[]}}\n\n'
    )
    client = build_v2_client(lambda request: Response(200, content=body))

    response = await client.chat(build_responses_request(stream=True))

    assert response.first_text == "kept"


@pytest.mark.asyncio
async def test_chat_stream_retries_a_failed_connect():
    attempts: list[int] = []

    def handler(request):
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectError("connection refused")

        return Response(200, content=SSE_BODY)

    client = build_v2_client(handler)

    response = await client.chat(build_responses_request(stream=True))

    assert len(attempts) == 2
    assert response.first_text == "Hello world"


@pytest.mark.asyncio
async def test_chat_stream_gives_up_after_repeated_connect_errors():
    def handler(request):
        raise ConnectError("connection refused")

    client = build_v2_client(handler)

    with pytest.raises(OpenAIV2HTTPClientError, match="transport error"):
        await client.chat(build_responses_request(stream=True))


@pytest.mark.asyncio
async def test_chat_stream_retries_a_drop_before_any_delta():
    attempts: list[int] = []

    def handler(request):
        attempts.append(1)
        if len(attempts) == 1:
            return Response(200, stream=FailingStream(b'data: {"type":"response.created"}\n\n'))

        return Response(200, content=SSE_BODY)

    client = build_v2_client(handler)

    response = await client.chat(build_responses_request(stream=True))

    assert len(attempts) == 2
    assert response.first_text == "Hello world"


@pytest.mark.asyncio
async def test_chat_stream_does_not_retry_a_drop_after_a_delta():
    attempts: list[int] = []

    def handler(request):
        attempts.append(1)
        return Response(
            200,
            stream=FailingStream(b'data: {"type":"response.output_text.delta","delta":"half"}\n\n'),
        )

    client = build_v2_client(handler)

    with pytest.raises(OpenAIV2HTTPClientError, match="transport error"):
        await client.chat(build_responses_request(stream=True))

    assert len(attempts) == 1


@pytest.mark.asyncio
async def test_chat_stream_does_not_retry_a_drop_after_a_reasoning_delta():
    attempts: list[int] = []

    def handler(request):
        attempts.append(1)
        return Response(
            200,
            stream=FailingStream(
                b'data: {"type":"response.reasoning_summary_text.delta","delta":"thinking"}\n\n'
            ),
        )

    client = build_v2_client(handler)

    with pytest.raises(OpenAIV2HTTPClientError, match="transport error"):
        await client.chat(build_responses_request(stream=True))

    assert len(attempts) == 1


@pytest.mark.asyncio
async def test_chat_stream_survives_an_error_after_completion():
    def handler(request):
        return Response(
            200,
            stream=FailingStream(
                b'data: {"type":"response.output_text.delta","delta":"whole"}\n\n'
                b'data: {"type":"response.completed","response":{"status":"completed","output":[]}}\n\n'
            ),
        )

    client = build_v2_client(handler)

    response = await client.chat(build_responses_request(stream=True))

    assert response.first_text == "whole"


@pytest.mark.asyncio
async def test_chat_without_stream_still_uses_plain_response():
    body = {
        "usage": {"total_tokens": 10, "input_tokens": 4, "output_tokens": 6},
        "output": [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "plain"}]}
        ],
        "status": "completed",
    }
    captured: list[dict] = []

    def handler(request):
        captured.append(json.loads(request.content))
        return Response(200, json=body)

    client = build_v2_client(handler)

    response = await client.chat(build_responses_request())

    assert response.first_text == "plain"
    assert captured[0]["stream"] is False
