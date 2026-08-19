from ai_review.config import settings
from ai_review.libs.logger import get_logger
from ai_review.services.cost.types import CostServiceProtocol
from ai_review.services.diff.types import DiffServiceProtocol
from ai_review.services.git.types import GitServiceProtocol
from ai_review.services.hook import hook
from ai_review.services.policy.types import PolicyServiceProtocol
from ai_review.services.prompt.adapter import build_prompt_context_from_review_info
from ai_review.services.prompt.types import PromptServiceProtocol
from ai_review.services.review.gateway.types import ReviewLLMGatewayProtocol, ReviewCommentGatewayProtocol
from ai_review.services.review.internal.summary.types import SummaryCommentServiceProtocol
from ai_review.services.review.runner.outcome import ReviewOutcome
from ai_review.services.review.runner.types import ReviewRunnerProtocol
from ai_review.services.vcs.types import VCSClientProtocol

logger = get_logger("SUMMARY_REVIEW_RUNNER")


class SummaryReviewRunner(ReviewRunnerProtocol):
    def __init__(
            self,
            vcs: VCSClientProtocol,
            git: GitServiceProtocol,
            diff: DiffServiceProtocol,
            cost: CostServiceProtocol,
            prompt: PromptServiceProtocol,
            policy: PolicyServiceProtocol,
            summary_comment: SummaryCommentServiceProtocol,
            review_llm_gateway: ReviewLLMGatewayProtocol,
            review_comment_gateway: ReviewCommentGatewayProtocol,
    ):
        self.vcs = vcs
        self.git = git
        self.diff = diff
        self.cost = cost
        self.prompt = prompt
        self.policy = policy
        self.summary_comment = summary_comment
        self.review_llm_gateway = review_llm_gateway
        self.review_comment_gateway = review_comment_gateway

    async def _build_prior_feedback(self) -> str | None:
        tag = settings.review.summary_tag
        threads = await self.vcs.get_general_threads()
        blocks: list[str] = []
        for thread in threads:
            if not thread.comments:
                continue
            root = thread.comments[0]
            if tag not in (root.body or ""):
                continue
            replies = [c for c in thread.comments[1:] if (c.body or "").strip()]
            if not replies:
                continue
            responses = "\n".join(
                f"- {c.author.username or c.author.name or 'user'}: {c.body.strip()}"
                for c in replies
            )
            blocks.append(
                "### A previous AI review\n"
                f"{root.body.strip()}\n\n"
                "### Team responses to that review (authoritative feedback — do not repeat "
                "findings that were refuted, and apply accepted corrections)\n"
                f"{responses}"
            )
        if not blocks:
            return None
        return "\n\n---\n\n".join(blocks)

    async def run(self) -> ReviewOutcome:
        await hook.emit_summary_review_start()

        comments = await self.review_comment_gateway.get_summary_comments()
        prior_feedback: str | None = None
        if comments:
            if not settings.review.summary_feedback_loop:
                logger.info(f"Detected {len(comments)} existing AI summary comments, skipping summary review")
                return ReviewOutcome.SKIPPED
            prior_feedback = await self._build_prior_feedback()
            logger.info(
                f"Feedback loop enabled: {len(comments)} prior summary comment(s); re-reviewing with "
                f"prior feedback in context (history kept, old summaries not deleted)"
            )

        review_info = await self.vcs.get_review_info()
        changed_files = self.policy.apply_for_files(review_info.changed_files)
        if not changed_files:
            logger.info("No files to review for summary")
            return ReviewOutcome.SKIPPED

        logger.info(f"Starting summary review: {len(changed_files)} files changed")

        rendered_files = self.diff.render_files(
            git=self.git,
            files=changed_files,
            base_sha=review_info.base_sha,
            head_sha=review_info.head_sha,
        )
        prompt_context = build_prompt_context_from_review_info(review_info)
        prompt = self.prompt.build_summary_request(rendered_files, prompt_context, prior_feedback=prior_feedback)
        prompt_system = self.prompt.build_system_summary_request(prompt_context)
        prompt_result = await self.review_llm_gateway.ask(prompt, prompt_system)

        summary = self.summary_comment.parse_model_output(prompt_result)
        if not summary.text.strip():
            logger.warning("Summary LLM output was empty, skipping comment")
            return ReviewOutcome.EMPTY

        logger.info(f"Posting summary review comment ({len(summary.text)} chars)")
        await self.review_comment_gateway.process_summary_comment(summary)
        await hook.emit_summary_review_complete(self.cost.aggregate())
        return ReviewOutcome.POSTED
