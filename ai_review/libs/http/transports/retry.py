import asyncio
from http import HTTPStatus
from typing import TYPE_CHECKING

from httpx import AsyncBaseTransport, ConnectError, ConnectTimeout, Request, Response

if TYPE_CHECKING:
    from loguru import Logger

TRANSPORT_ERROR_TYPES = (ConnectError, ConnectTimeout)


class RetryTransport(AsyncBaseTransport):
    def __init__(
            self,
            logger: "Logger",
            transport: AsyncBaseTransport,
            max_retries: int = 5,
            retry_delay: float = 0.5,
            retry_status_codes: tuple[HTTPStatus, ...] = (
                    HTTPStatus.BAD_GATEWAY,
                    HTTPStatus.GATEWAY_TIMEOUT,
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    HTTPStatus.INTERNAL_SERVER_ERROR,
            ),
            retry_transport_errors: bool = False,
            transport_error_retries: int = 3,
    ):
        self.logger = logger
        self.transport = transport
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_status_codes = retry_status_codes
        self.retry_transport_errors = retry_transport_errors
        self.transport_error_retries = transport_error_retries

    def _delay_for(self, attempt_index: int) -> float:
        return self.retry_delay * 2 ** attempt_index

    async def _handle_with_transport_error_retries(self, request: Request) -> Response:
        last_error: Exception | None = None
        for attempt in range(self.transport_error_retries):
            try:
                return await self.transport.handle_async_request(request)
            except TRANSPORT_ERROR_TYPES as error:
                last_error = error

                if attempt == self.transport_error_retries - 1:
                    break

                delay = self._delay_for(attempt)
                self.logger.warning(
                    f"Attempt {attempt + 1}/{self.transport_error_retries} failed "
                    f"with {error!r} for {request.method} {request.url}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)

        if last_error is None:
            raise RuntimeError(f"RetryTransport made no attempt for {request.method} {request.url}")

        self.logger.error(
            f"All {self.transport_error_retries} attempts failed for "
            f"{request.method} {request.url} (last error={last_error!r})"
        )

        raise last_error

    async def handle_async_request(self, request: Request) -> Response:
        last_response: Response | None = None
        for attempt in range(self.max_retries):
            if self.retry_transport_errors:
                last_response = await self._handle_with_transport_error_retries(request)
            else:
                last_response = await self.transport.handle_async_request(request)

            if last_response.status_code not in self.retry_status_codes:
                return last_response

            if attempt == self.max_retries - 1:
                break

            delay = self._delay_for(attempt)
            self.logger.warning(
                f"Attempt {attempt + 1}/{self.max_retries} failed "
                f"with status={last_response.status_code} for {request.method} {request.url}. "
                f"Retrying in {delay:.1f}s..."
            )

            await last_response.aclose()
            await asyncio.sleep(delay)

        if last_response is None:
            raise RuntimeError(f"RetryTransport made no attempt for {request.method} {request.url}")

        self.logger.error(
            f"All {self.max_retries} attempts failed for "
            f"{request.method} {request.url} (last status={last_response.status_code})"
        )

        return last_response
