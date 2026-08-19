from types import SimpleNamespace

import pytest
from httpx import (
    AsyncBaseTransport,
    AsyncByteStream,
    AsyncClient,
    ConnectError,
    ConnectTimeout,
    MockTransport,
    ReadTimeout,
    Request,
    Response,
)
from httpx._exceptions import StreamClosed

from ai_review.libs.http.transports import retry as retry_module
from ai_review.libs.http.transports.retry import RetryTransport
from ai_review.libs.logger import get_logger


class ByteStream(AsyncByteStream):
    def __init__(self, payload: bytes):
        self.payload = payload
        self.closed = False

    async def __aiter__(self):
        if self.closed:
            raise StreamClosed()

        yield self.payload

    async def aclose(self) -> None:
        self.closed = True


class ScriptedTransport(AsyncBaseTransport):
    def __init__(self, outcomes: list):
        self.outcomes = list(outcomes)
        self.calls = 0

    async def handle_async_request(self, request: Request) -> Response:
        outcome = self.outcomes[self.calls]
        self.calls += 1

        if isinstance(outcome, Exception):
            raise outcome

        return outcome


def patch_sleep(monkeypatch) -> list[float]:
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(retry_module, "asyncio", SimpleNamespace(sleep=fake_sleep))

    return delays


def build_retry_client(
        handler,
        max_retries: int = 2,
        retry_delay: float = 0,
        retry_transport_errors: bool = False,
        transport_error_retries: int = 3,
) -> AsyncClient:
    transport = RetryTransport(
        logger=get_logger("TEST_RETRY"),
        transport=MockTransport(handler),
        max_retries=max_retries,
        retry_delay=retry_delay,
        retry_transport_errors=retry_transport_errors,
        transport_error_retries=transport_error_retries,
    )
    return AsyncClient(transport=transport, base_url="https://example.com")


@pytest.mark.asyncio
async def test_retry_transport_returns_a_readable_response_after_exhausting_retries():
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


@pytest.mark.asyncio
async def test_retry_transport_uses_exponential_backoff_between_status_retries(monkeypatch):
    delays = patch_sleep(monkeypatch)

    def handler(request):
        return Response(500, stream=ByteStream(b"boom"))

    async with build_retry_client(handler, max_retries=4, retry_delay=0.01) as client:
        response = await client.post("/thing")

        assert response.status_code == 500

    assert delays == [0.01, 0.02, 0.04]


@pytest.mark.asyncio
async def test_default_retry_transport_does_not_retry_transport_errors():
    transport = RetryTransport(
        logger=get_logger("TEST_RETRY"),
        transport=ScriptedTransport([ConnectError("boom")]),
    )
    client = AsyncClient(transport=transport, base_url="https://example.com")

    async with client:
        with pytest.raises(ConnectError):
            await client.get("/thing")

    assert transport.transport.calls == 1


@pytest.mark.asyncio
async def test_retry_transport_retries_connect_errors_until_success(monkeypatch):
    delays = patch_sleep(monkeypatch)
    scripted = ScriptedTransport([ConnectError("boom"), ConnectError("boom"), Response(200, json={"ok": True})])
    transport = RetryTransport(
        logger=get_logger("TEST_RETRY"),
        transport=scripted,
        retry_transport_errors=True,
        transport_error_retries=3,
    )
    client = AsyncClient(transport=transport, base_url="https://example.com")

    async with client:
        response = await client.get("/thing")

        assert response.json() == {"ok": True}

    assert scripted.calls == 3
    assert delays == [0.5, 1.0]


@pytest.mark.asyncio
async def test_retry_transport_reraises_connect_error_after_exhausting_transport_error_retries(monkeypatch):
    patch_sleep(monkeypatch)
    scripted = ScriptedTransport([ConnectError("boom")] * 3)
    transport = RetryTransport(
        logger=get_logger("TEST_RETRY"),
        transport=scripted,
        retry_transport_errors=True,
        transport_error_retries=3,
    )
    client = AsyncClient(transport=transport, base_url="https://example.com")

    async with client:
        with pytest.raises(ConnectError):
            await client.get("/thing")

    assert scripted.calls == 3


@pytest.mark.asyncio
async def test_retry_transport_retries_connect_timeout_when_enabled(monkeypatch):
    patch_sleep(monkeypatch)
    scripted = ScriptedTransport([ConnectTimeout("boom"), Response(200, json={"ok": True})])
    transport = RetryTransport(
        logger=get_logger("TEST_RETRY"),
        transport=scripted,
        retry_transport_errors=True,
    )
    client = AsyncClient(transport=transport, base_url="https://example.com")

    async with client:
        response = await client.get("/thing")

        assert response.json() == {"ok": True}

    assert scripted.calls == 2


@pytest.mark.asyncio
async def test_retry_transport_does_not_retry_read_timeout_even_when_enabled(monkeypatch):
    patch_sleep(monkeypatch)
    scripted = ScriptedTransport([ReadTimeout("boom"), Response(200, json={"ok": True})])
    transport = RetryTransport(
        logger=get_logger("TEST_RETRY"),
        transport=scripted,
        retry_transport_errors=True,
    )
    client = AsyncClient(transport=transport, base_url="https://example.com")

    async with client:
        with pytest.raises(ReadTimeout):
            await client.get("/thing")

    assert scripted.calls == 1
