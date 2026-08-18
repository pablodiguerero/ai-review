import json

import pytest
from httpx import AsyncByteStream, AsyncClient, ConnectError, MockTransport, ReadError, Response

from ai_review.clients.openai.v1.client import (
    get_openai_v1_http_client,
    OpenAIV1HTTPClient,
    OpenAIV1HTTPClientError,
)
from ai_review.clients.openai.v1.schema import (
    OpenAIChatRequestSchema,
    OpenAIMessageSchema,
    OpenAIStreamOptionsSchema,
)


@pytest.mark.usefixtures('openai_v1_http_client_config')
def test_get_openai_v1_http_client_builds_ok():
    openai_http_client = get_openai_v1_http_client()

    assert isinstance(openai_http_client, OpenAIV1HTTPClient)
    assert isinstance(openai_http_client.client, AsyncClient)


SSE_BODY = (
    b'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}],"usage":null}\n\n'
    b'data: {"choices":[{"delta":{"content":"Hello"}}],"usage":null}\n\n'
    b'data: {"choices":[{"delta":{"content":" world"}}],"usage":null}\n\n'
    b'data: {"choices":[],"usage":{"total_tokens":44,"prompt_tokens":6,"completion_tokens":38}}\n\n'
    b'data: [DONE]\n\n'
    b'data: {"choices":[],"cost":"0.00001092"}\n\n'
)


class FailingStream(AsyncByteStream):
    """Yields a prefix, then dies the way a dropped connection does."""

    def __init__(self, prefix: bytes):
        self.prefix = prefix

    async def __aiter__(self):
        yield self.prefix
        raise ReadError("connection dropped mid-body")


def build_stream_client(handler) -> OpenAIV1HTTPClient:
    return OpenAIV1HTTPClient(
        client=AsyncClient(base_url="https://api.openai.com/v1", transport=MockTransport(handler))
    )


def build_chat_request(**kwargs) -> OpenAIChatRequestSchema:
    return OpenAIChatRequestSchema(
        model="gpt-4o-mini",
        messages=[OpenAIMessageSchema(role="user", content="hi")],
        **kwargs
    )


@pytest.mark.asyncio
async def test_chat_stream_collects_content_and_usage():
    client = build_stream_client(lambda request: Response(200, content=SSE_BODY))

    response = await client.chat(build_chat_request(stream=True))

    assert response.first_text == "Hello world"
    assert response.usage.total_tokens == 44
    assert response.usage.prompt_tokens == 6
    assert response.usage.completion_tokens == 38


@pytest.mark.asyncio
async def test_chat_stream_sends_stream_flag():
    captured: list[dict] = []

    def handler(request):
        captured.append(json.loads(request.content))
        return Response(200, content=SSE_BODY)

    client = build_stream_client(handler)

    await client.chat(build_chat_request(stream=True, stream_options=OpenAIStreamOptionsSchema()))

    assert captured[0]["stream"] is True
    assert captured[0]["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_chat_stream_ignores_content_after_done():
    body = (
        b'data: {"choices":[{"delta":{"content":"kept"}}]}\n\n'
        b'data: [DONE]\n\n'
        b'data: {"choices":[{"delta":{"content":"dropped"}}]}\n\n'
    )
    client = build_stream_client(lambda request: Response(200, content=body))

    response = await client.chat(build_chat_request(stream=True))

    assert response.first_text == "kept"


@pytest.mark.asyncio
async def test_chat_stream_keeps_only_the_first_choice():
    body = (
        b'data: {"choices":[{"index":0,"delta":{"content":"first"}},'
        b'{"index":1,"delta":{"content":"second"}}]}\n\n'
        b'data: [DONE]\n\n'
    )
    client = build_stream_client(lambda request: Response(200, content=body))

    response = await client.chat(build_chat_request(stream=True))

    assert response.first_text == "first"


@pytest.mark.asyncio
async def test_chat_stream_keeps_a_usage_block_without_a_total():
    body = (
        b'data: {"choices":[{"delta":{"content":"kept"}}],'
        b'"usage":{"prompt_tokens":1,"completion_tokens":2}}\n\n'
        b'data: [DONE]\n\n'
    )
    client = build_stream_client(lambda request: Response(200, content=body))

    response = await client.chat(build_chat_request(stream=True))

    assert response.first_text == "kept"
    assert response.usage.prompt_tokens == 1
    assert response.usage.completion_tokens == 2
    assert response.usage.total_tokens == 3


@pytest.mark.asyncio
async def test_chat_stream_raises_when_a_running_usage_total_precedes_a_drop():
    # Usage riding along a CONTENT chunk is a running total, not an ending:
    # trusting it would return half an answer as the finished review.
    def handler(request):
        return Response(
            200,
            stream=FailingStream(
                b'data: {"choices":[{"delta":{"content":"half"}}],'
                b'"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}}\n\n'
            ),
        )

    client = build_stream_client(handler)

    with pytest.raises(OpenAIV1HTTPClientError, match="transport error"):
        await client.chat(build_chat_request(stream=True))


@pytest.mark.asyncio
async def test_chat_stream_accepts_a_usage_frame_as_the_only_completion_signal():
    # grok on the zen gateway: no [DONE], finish_reason always null, usage last.
    body = (
        b'data: {"choices":[{"index":0,"delta":{"content":"whole"},"finish_reason":null}]}\n\n'
        b'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}\n\n'
        b'data: {"choices":[],"cost":"0.00066200"}\n\n'
    )
    client = build_stream_client(lambda request: Response(200, content=body))

    response = await client.chat(build_chat_request(stream=True))

    assert response.first_text == "whole"
    assert response.usage.total_tokens == 12


@pytest.mark.asyncio
async def test_chat_stream_raises_on_truncated_stream():
    body = b'data: {"choices":[{"delta":{"content":"half an ans"}}]}\n\n'
    client = build_stream_client(lambda request: Response(200, content=body))

    with pytest.raises(OpenAIV1HTTPClientError, match="completion signal"):
        await client.chat(build_chat_request(stream=True))


@pytest.mark.asyncio
async def test_chat_stream_raises_on_error_frame():
    body = b'data: {"error":{"message":"Upstream request failed"}}\n\ndata: [DONE]\n\n'
    client = build_stream_client(lambda request: Response(200, content=body))

    with pytest.raises(OpenAIV1HTTPClientError, match="Upstream request failed"):
        await client.chat(build_chat_request(stream=True))


@pytest.mark.asyncio
async def test_chat_stream_returns_empty_text_when_no_content_arrived():
    # The agent loop retries empty answers itself, so this must not raise.
    body = b'data: {"choices":[]}\n\ndata: [DONE]\n\n'
    client = build_stream_client(lambda request: Response(200, content=body))

    response = await client.chat(build_chat_request(stream=True))

    assert response.first_text == ""


@pytest.mark.asyncio
async def test_chat_stream_accepts_a_finish_reason_without_done():
    body = b'data: {"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}]}\n\n'
    client = build_stream_client(lambda request: Response(200, content=body))

    response = await client.chat(build_chat_request(stream=True))

    assert response.first_text == "done"


@pytest.mark.asyncio
async def test_chat_stream_raises_when_truncated_by_max_tokens():
    body = (
        b'data: {"choices":[{"delta":{"content":"half"},"finish_reason":"length"}]}\n\n'
        b'data: [DONE]\n\n'
    )
    client = build_stream_client(lambda request: Response(200, content=body))

    with pytest.raises(OpenAIV1HTTPClientError, match="max_tokens"):
        await client.chat(build_chat_request(stream=True))


@pytest.mark.asyncio
async def test_chat_stream_keeps_the_first_meaningful_usage_block():
    body = (
        b'data: {"choices":[{"delta":{"content":"text"}}],'
        b'"usage":{"total_tokens":44,"prompt_tokens":6,"completion_tokens":38}}\n\n'
        b'data: {"choices":[],"usage":{"total_tokens":0,"prompt_tokens":0,"completion_tokens":0}}\n\n'
        b'data: [DONE]\n\n'
    )
    client = build_stream_client(lambda request: Response(200, content=body))

    response = await client.chat(build_chat_request(stream=True))

    assert response.usage.total_tokens == 44


@pytest.mark.asyncio
async def test_chat_stream_rejects_an_empty_finish_reason_as_completion():
    # Several gateways stamp "finish_reason": "" on every content chunk; reading
    # that as an ending would disarm the truncation guard for the whole stream.
    body = b'data: {"choices":[{"delta":{"content":"half"},"finish_reason":""}]}\n\n'
    client = build_stream_client(lambda request: Response(200, content=body))

    with pytest.raises(OpenAIV1HTTPClientError, match="completion signal"):
        await client.chat(build_chat_request(stream=True))


@pytest.mark.asyncio
async def test_chat_stream_retries_a_failed_connect():
    attempts: list[int] = []

    def handler(request):
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectError("connection refused")

        return Response(200, content=SSE_BODY)

    client = build_stream_client(handler)

    response = await client.chat(build_chat_request(stream=True))

    assert len(attempts) == 2
    assert response.first_text == "Hello world"


@pytest.mark.asyncio
async def test_chat_stream_gives_up_after_repeated_connect_errors():
    def handler(request):
        raise ConnectError("connection refused")

    client = build_stream_client(handler)

    with pytest.raises(OpenAIV1HTTPClientError, match="transport error"):
        await client.chat(build_chat_request(stream=True))


@pytest.mark.asyncio
async def test_chat_stream_retries_a_drop_after_a_prompt_only_usage_frame():
    # Nothing was generated yet, so the retry costs nothing extra.
    attempts: list[int] = []

    def handler(request):
        attempts.append(1)
        if len(attempts) == 1:
            return Response(
                200,
                stream=FailingStream(
                    b'data: {"choices":[],"usage":{"prompt_tokens":12000,"completion_tokens":0,"total_tokens":12000}}\n\n'
                ),
            )

        return Response(200, content=SSE_BODY)

    client = build_stream_client(handler)

    response = await client.chat(build_chat_request(stream=True))

    assert len(attempts) == 2
    assert response.first_text == "Hello world"


@pytest.mark.asyncio
async def test_chat_stream_raises_when_a_dropped_chunk_carried_content():
    body = (
        b'data: {"choices":[{"delta":{"content":"kept"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":["broken"]}}]}\n\n'
        b'data: [DONE]\n\n'
    )
    client = build_stream_client(lambda request: Response(200, content=body))

    with pytest.raises(OpenAIV1HTTPClientError, match="carried content"):
        await client.chat(build_chat_request(stream=True))


@pytest.mark.asyncio
async def test_chat_stream_does_not_retry_a_drop_after_reasoning():
    # Reasoning tokens are billed like output, so a retry here pays twice.
    attempts: list[int] = []

    def handler(request):
        attempts.append(1)
        return Response(
            200,
            stream=FailingStream(b'data: {"choices":[{"delta":{"reasoning":"thinking"}}]}\n\n'),
        )

    client = build_stream_client(handler)

    with pytest.raises(OpenAIV1HTTPClientError, match="transport error"):
        await client.chat(build_chat_request(stream=True))

    assert len(attempts) == 1


@pytest.mark.asyncio
async def test_chat_stream_survives_an_error_after_the_usage_frame():
    # The gateway's last frame arrived, so the answer is whole - an error while
    # closing the socket must not throw it away.
    def handler(request):
        return Response(
            200,
            stream=FailingStream(
                b'data: {"choices":[{"delta":{"content":"whole"},"finish_reason":"stop"}]}\n\n'
                b'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}\n\n'
            ),
        )

    client = build_stream_client(handler)

    response = await client.chat(build_chat_request(stream=True))

    assert response.first_text == "whole"


@pytest.mark.asyncio
async def test_chat_stream_retries_a_drop_before_any_content():
    attempts: list[int] = []

    def handler(request):
        attempts.append(1)
        if len(attempts) == 1:
            return Response(200, stream=FailingStream(b'data: {"choices":[]}\n\n'))

        return Response(200, content=SSE_BODY)

    client = build_stream_client(handler)

    response = await client.chat(build_chat_request(stream=True))

    assert len(attempts) == 2
    assert response.first_text == "Hello world"


@pytest.mark.asyncio
async def test_chat_stream_does_not_retry_a_drop_after_content():
    attempts: list[int] = []

    def handler(request):
        attempts.append(1)
        return Response(200, stream=FailingStream(b'data: {"choices":[{"delta":{"content":"half"}}]}\n\n'))

    client = build_stream_client(handler)

    with pytest.raises(OpenAIV1HTTPClientError, match="transport error"):
        await client.chat(build_chat_request(stream=True))

    assert len(attempts) == 1


@pytest.mark.asyncio
async def test_chat_stream_without_usage_chunk_defaults_to_zero():
    body = b'data: {"choices":[{"delta":{"content":"only text"}}]}\n\ndata: [DONE]\n\n'
    client = build_stream_client(lambda request: Response(200, content=body))

    response = await client.chat(build_chat_request(stream=True))

    assert response.first_text == "only text"
    assert response.usage.total_tokens == 0


@pytest.mark.asyncio
async def test_chat_stream_skips_malformed_chunks():
    body = (
        b'data: not-json\n\n'
        b'data: {"choices":[{"delta":{"content":"kept"}}]}\n\n'
        b': keep-alive comment\n\n'
        b'data: [DONE]\n\n'
    )
    client = build_stream_client(lambda request: Response(200, content=body))

    response = await client.chat(build_chat_request(stream=True))

    assert response.first_text == "kept"


@pytest.mark.asyncio
async def test_chat_stream_raises_on_error_status():
    client = build_stream_client(lambda request: Response(402, content=b'{"error":"no credits"}'))

    with pytest.raises(OpenAIV1HTTPClientError) as error:
        await client.chat(build_chat_request(stream=True))

    assert error.value.status_code == 402
    assert "no credits" in error.value.details


@pytest.mark.asyncio
async def test_chat_without_stream_uses_plain_response():
    body = {
        "usage": {"total_tokens": 10, "prompt_tokens": 4, "completion_tokens": 6},
        "choices": [{"message": {"role": "assistant", "content": "plain"}}],
    }
    client = build_stream_client(lambda request: Response(200, json=body))

    response = await client.chat(build_chat_request())

    assert response.first_text == "plain"
    assert response.usage.total_tokens == 10
