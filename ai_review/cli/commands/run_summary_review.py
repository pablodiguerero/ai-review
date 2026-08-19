from ai_review.services.review.runner.outcome import ReviewOutcome
from ai_review.services.review.service import ReviewService


async def run_summary_review_command() -> ReviewOutcome:
    review_service = ReviewService()
    outcome = await review_service.run_summary_review()
    review_service.report_total_cost()
    return outcome
