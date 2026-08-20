import pytest

from ai_review.config import settings
from ai_review.services.diff.schema import DiffFileSchema
from ai_review.services.review.internal.summary.schema import SummaryCommentSchema
from ai_review.services.review.runner.outcome import ReviewOutcome
from ai_review.services.review.runner.summary import SummaryReviewRunner, build_summary_checkpoint_key
from ai_review.services.vcs.types import ReviewCommentSchema, ReviewInfoSchema
from ai_review.tests.fixtures.services.cost import FakeCostService
from ai_review.tests.fixtures.services.diff import FakeDiffService
from ai_review.tests.fixtures.services.policy import FakePolicyService
from ai_review.tests.fixtures.services.prompt import FakePromptService
from ai_review.tests.fixtures.services.review.gateway.review_comment_gateway import FakeReviewCommentGateway
from ai_review.tests.fixtures.services.review.gateway.review_direct_llm_gateway import FakeReviewDirectLLMGateway
from ai_review.tests.fixtures.services.review.internal.summary import FakeSummaryCommentService
from ai_review.tests.fixtures.services.vcs import FakeVCSClient


@pytest.mark.asyncio
async def test_run_happy_path(
        summary_review_runner: SummaryReviewRunner,
        fake_vcs_client: FakeVCSClient,
        fake_diff_service: FakeDiffService,
        fake_cost_service: FakeCostService,
        fake_prompt_service: FakePromptService,
        fake_policy_service: FakePolicyService,
        fake_review_comment_gateway: FakeReviewCommentGateway,
        fake_review_direct_llm_gateway: FakeReviewDirectLLMGateway,
):
    fake_review_comment_gateway.responses["get_summary_comments"] = []

    outcome = await summary_review_runner.run()

    assert outcome == ReviewOutcome.POSTED

    vcs_calls = [call[0] for call in fake_vcs_client.calls]
    assert "get_review_info" in vcs_calls

    assert any(call[0] == "render_batches" for call in fake_diff_service.calls)
    assert any(call[0] == "apply_for_files" for call in fake_policy_service.calls)
    assert any(call[0] == "build_summary_request" for call in fake_prompt_service.calls)
    assert any(call[0] == "ask" for call in fake_review_direct_llm_gateway.calls)
    assert any(call[0] == "process_summary_comment" for call in fake_review_comment_gateway.calls)

    process_call = next(
        call for call in fake_review_comment_gateway.calls
        if call[0] == "process_summary_comment"
    )
    assert process_call[1]["previous"] == []

    assert any(call[0] == "aggregate" for call in fake_cost_service.calls)


@pytest.mark.asyncio
async def test_run_passes_checkpoint_key_built_from_review_info_to_ask(
        summary_review_runner: SummaryReviewRunner,
        fake_vcs_client: FakeVCSClient,
        fake_review_comment_gateway: FakeReviewCommentGateway,
        fake_review_direct_llm_gateway: FakeReviewDirectLLMGateway,
):
    fake_review_comment_gateway.responses["get_summary_comments"] = []
    review_info = ReviewInfoSchema(changed_files=["file.py"], base_sha="A", head_sha="deadbeef")
    fake_vcs_client.responses["get_review_info"] = review_info

    await summary_review_runner.run()

    ask_call = next(call for call in fake_review_direct_llm_gateway.calls if call[0] == "ask")
    expected_key = build_summary_checkpoint_key(review_info)
    assert ask_call[1]["checkpoint_key"] == expected_key
    assert "deadbeef" not in expected_key
    assert ask_call[1]["checkpoint_head_sha"] == "deadbeef"


def test_checkpoint_key_is_stable_across_different_head_shas_for_the_same_mr_model_and_tag() -> None:
    first = build_summary_checkpoint_key(ReviewInfoSchema(changed_files=[], base_sha="A", head_sha="sha1"))
    second = build_summary_checkpoint_key(ReviewInfoSchema(changed_files=[], base_sha="A", head_sha="sha2"))

    assert first == second


@pytest.mark.asyncio
async def test_run_passes_existing_comments_as_previous_when_feedback_loop_enabled(
        monkeypatch: pytest.MonkeyPatch,
        summary_review_runner: SummaryReviewRunner,
        fake_review_comment_gateway: FakeReviewCommentGateway,
):
    monkeypatch.setattr(settings.review, "summary_feedback_loop", True)
    existing = [ReviewCommentSchema(id="1", body="#ai-review-summary existing")]
    fake_review_comment_gateway.responses["get_summary_comments"] = existing

    outcome = await summary_review_runner.run()

    assert outcome == ReviewOutcome.POSTED
    process_call = next(
        call for call in fake_review_comment_gateway.calls
        if call[0] == "process_summary_comment"
    )
    assert process_call[1]["previous"] == existing


@pytest.mark.asyncio
async def test_run_skips_when_existing_summary_comments(
        summary_review_runner: SummaryReviewRunner,
        fake_vcs_client: FakeVCSClient,
        fake_review_comment_gateway: FakeReviewCommentGateway,
        fake_review_direct_llm_gateway: FakeReviewDirectLLMGateway,
):
    fake_review_comment_gateway.responses["get_summary_comments"] = [
        ReviewCommentSchema(id="1", body="#ai-review-summary existing"),
    ]

    outcome = await summary_review_runner.run()

    assert outcome == ReviewOutcome.SKIPPED
    vcs_calls = [call[0] for call in fake_vcs_client.calls]
    assert vcs_calls == []
    assert not any(call[0] == "ask" for call in fake_review_direct_llm_gateway.calls)


@pytest.mark.asyncio
async def test_run_skips_when_no_changed_files(
        summary_review_runner: SummaryReviewRunner,
        fake_vcs_client: FakeVCSClient,
        fake_policy_service: FakePolicyService,
        fake_review_comment_gateway: FakeReviewCommentGateway,
):
    fake_policy_service.responses["apply_for_files"] = []
    fake_review_comment_gateway.responses["get_summary_comments"] = []

    outcome = await summary_review_runner.run()

    assert outcome == ReviewOutcome.SKIPPED
    vcs_calls = [call[0] for call in fake_vcs_client.calls]
    assert "get_review_info" in vcs_calls
    assert any(call[0] == "apply_for_files" for call in fake_policy_service.calls)


@pytest.mark.asyncio
async def test_run_skips_when_empty_summary_from_llm(
        summary_review_runner: SummaryReviewRunner,
        fake_review_comment_gateway: FakeReviewCommentGateway,
        fake_summary_comment_service: FakeSummaryCommentService,
        fake_review_direct_llm_gateway: FakeReviewDirectLLMGateway,
):
    fake_review_comment_gateway.responses["get_summary_comments"] = []
    fake_summary_comment_service.responses["parse_model_output"] = SummaryCommentSchema(text="")

    outcome = await summary_review_runner.run()

    assert outcome == ReviewOutcome.EMPTY
    assert any(call[0] == "ask" for call in fake_review_direct_llm_gateway.calls)
    assert not any(call[0] == "process_summary_comment" for call in fake_review_comment_gateway.calls)


@pytest.mark.asyncio
async def test_run_batched_posts_one_consolidated_comment_with_min_score_and_per_batch_checkpoint_keys(
        summary_review_runner: SummaryReviewRunner,
        fake_vcs_client: FakeVCSClient,
        fake_diff_service: FakeDiffService,
        fake_review_comment_gateway: FakeReviewCommentGateway,
        fake_review_direct_llm_gateway: FakeReviewDirectLLMGateway,
        fake_summary_comment_service: FakeSummaryCommentService,
):
    fake_review_comment_gateway.responses["get_summary_comments"] = []
    review_info = ReviewInfoSchema(changed_files=["a.py", "b.py"], base_sha="A", head_sha="deadbeef")
    fake_vcs_client.responses["get_review_info"] = review_info
    fake_diff_service.responses["render_batches"] = [
        [DiffFileSchema(file="a.py", diff="diff a")],
        [DiffFileSchema(file="b.py", diff="diff b")],
    ]
    fake_review_direct_llm_gateway.responses["ask"] = [
        "Findings for a.\n\nOverall score: 9",
        "Findings for b.\n\nOverall score: 6",
    ]

    outcome = await summary_review_runner.run()

    assert outcome == ReviewOutcome.POSTED

    ask_calls = [call for call in fake_review_direct_llm_gateway.calls if call[0] == "ask"]
    assert len(ask_calls) == 2
    base_key = build_summary_checkpoint_key(review_info)
    assert ask_calls[0][1]["checkpoint_key"] == f"{base_key}:b1"
    assert ask_calls[1][1]["checkpoint_key"] == f"{base_key}:b2"
    assert ask_calls[0][1]["checkpoint_head_sha"] == "deadbeef"
    assert ask_calls[1][1]["checkpoint_head_sha"] == "deadbeef"
    assert "This part covers these files: a.py" in ask_calls[0][1]["prompt"]
    assert "This part covers these files: b.py" in ask_calls[1][1]["prompt"]

    parse_call = next(
        call for call in fake_summary_comment_service.calls
        if call[0] == "parse_model_output"
    )
    consolidated_text = parse_call[1]["output"]
    assert "Batched review: 2 parts covering 2 of 2 changed files." in consolidated_text
    assert "## Part 1/2 — 1 files" in consolidated_text
    assert "## Part 2/2 — 1 files" in consolidated_text
    assert "Findings for a." in consolidated_text
    assert "Findings for b." in consolidated_text
    assert consolidated_text.strip().endswith("Overall score: 6.0")

    assert any(call[0] == "process_summary_comment" for call in fake_review_comment_gateway.calls)


@pytest.mark.asyncio
async def test_run_batched_treats_one_failed_part_as_skipped_but_still_posts(
        summary_review_runner: SummaryReviewRunner,
        fake_review_comment_gateway: FakeReviewCommentGateway,
        fake_diff_service: FakeDiffService,
        fake_review_direct_llm_gateway: FakeReviewDirectLLMGateway,
        fake_summary_comment_service: FakeSummaryCommentService,
):
    fake_review_comment_gateway.responses["get_summary_comments"] = []
    fake_diff_service.responses["render_batches"] = [
        [DiffFileSchema(file="a.py", diff="diff a")],
        [DiffFileSchema(file="b.py", diff="diff b")],
    ]
    fake_review_direct_llm_gateway.responses["ask"] = ["", "Findings for b.\n\nOverall score: 6"]

    outcome = await summary_review_runner.run()

    assert outcome == ReviewOutcome.POSTED
    parse_call = next(
        call for call in fake_summary_comment_service.calls
        if call[0] == "parse_model_output"
    )
    consolidated_text = parse_call[1]["output"]
    assert "## Part 1/2 — no result (skipped)" in consolidated_text
    assert "## Part 2/2 — 1 files" in consolidated_text
    assert consolidated_text.strip().endswith("Overall score: 6.0")


@pytest.mark.asyncio
async def test_run_batched_all_parts_empty_returns_empty_without_posting(
        summary_review_runner: SummaryReviewRunner,
        fake_review_comment_gateway: FakeReviewCommentGateway,
        fake_diff_service: FakeDiffService,
        fake_review_direct_llm_gateway: FakeReviewDirectLLMGateway,
):
    fake_review_comment_gateway.responses["get_summary_comments"] = []
    fake_diff_service.responses["render_batches"] = [
        [DiffFileSchema(file="a.py", diff="diff a")],
        [DiffFileSchema(file="b.py", diff="diff b")],
    ]
    fake_review_direct_llm_gateway.responses["ask"] = ["", ""]

    outcome = await summary_review_runner.run()

    assert outcome == ReviewOutcome.EMPTY
    assert not any(call[0] == "process_summary_comment" for call in fake_review_comment_gateway.calls)
