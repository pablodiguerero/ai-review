from typing import Protocol

from ai_review.services.review.runner.outcome import ReviewOutcome


class ReviewRunnerProtocol(Protocol):
    async def run(self) -> ReviewOutcome | None:
        ...
