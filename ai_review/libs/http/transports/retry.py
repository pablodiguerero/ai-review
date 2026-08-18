import asyncio
from http import HTTPStatus
from typing import TYPE_CHECKING

from httpx import Request, Response, AsyncBaseTransport

if TYPE_CHECKING:
    from loguru import Logger


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
            )
    ):
        self.logger = logger
        self.transport = transport
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_status_codes = retry_status_codes

    async def handle_async_request(self, request: Request) -> Response:
        last_response: Response | None = None
        for attempt in range(self.max_retries):
            last_response = await self.transport.handle_async_request(request)
            if last_response.status_code not in self.retry_status_codes:
                return last_response

            if attempt == self.max_retries - 1:
                break

            self.logger.warning(
                f"Attempt {attempt + 1}/{self.max_retries} failed "
                f"with status={last_response.status_code} for {request.method} {request.url}. "
                f"Retrying in {self.retry_delay:.1f}s..."
            )

            # The discarded response still owns its connection; a streaming one
            # would hold it out of the pool for the rest of the run. The response
            # returned below is deliberately left open for the caller to read.
            await last_response.aclose()
            await asyncio.sleep(self.retry_delay)

        if last_response is None:
            raise RuntimeError(f"RetryTransport made no attempt for {request.method} {request.url}")

        self.logger.error(
            f"All {self.max_retries} attempts failed for "
            f"{request.method} {request.url} (last status={last_response.status_code})"
        )

        return last_response
