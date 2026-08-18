import pytest
from httpx import AsyncByteStream, AsyncClient, MockTransport, Response
from httpx._exceptions import StreamClosed

from ai_review.libs.http.transports.retry import RetryTransport
from ai_review.libs.logger import get_logger


class ByteStream(AsyncByteStream):
    """Behaves like a socket: once closed, reading it fails."""

    def __init__(self, payload: bytes):
        self.payload = payload
        self.closed = False

    async def __aiter__(self):
        if self.closed:
            raise StreamClosed()

        yield self.payload

    async def aclose(self) -> None:
        self.closed = True


def build_retry_client(handler, max_retries: int = 2) -> AsyncClient:
    transport = RetryTransport(
        logger=get_logger("TEST_RETRY"),
        transport=MockTransport(handler),
        max_retries=max_retries,
        retry_delay=0,
    )
    return AsyncClient(transport=transport, base_url="https://example.com")


@pytest.mark.asyncio
async def test_retry_transport_returns_a_readable_response_after_exhausting_retries():
    # The returned response must still be readable: callers read the body to
    # build their own error, and closing it surfaces StreamClosed instead.
    def handler(request):
        return Response(500, stream=ByteStream(b'{"error":"upstream"}'))

    async with build_retry_client(handler) as client:
        response = await client.post("/thing")

        assert response.status_code == 500
        assert response.text == '{"error":"upstream"}'


@pytest.mark.asyncio
async def test_retry_transport_retries_until_success():
    attempts: list[int] = []

    def handler(request):
        attempts.append(1)
        if len(attempts) == 1:
            return Response(500, stream=ByteStream(b"boom"))

        return Response(200, json={"ok": True})

    async with build_retry_client(handler) as client:
        response = await client.post("/thing")

        assert len(attempts) == 2
        assert response.json() == {"ok": True}
