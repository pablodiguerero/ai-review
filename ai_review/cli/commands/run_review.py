from ai_review.services.review.runner.outcome import ReviewOutcome
from ai_review.services.review.service import ReviewService


async def run_review_command() -> tuple[ReviewOutcome, ReviewOutcome]:
    review_service = ReviewService()
    inline_outcome = await review_service.run_inline_review()
    summary_outcome = await review_service.run_summary_review()
    review_service.report_total_cost()
    return inline_outcome, summary_outcome
