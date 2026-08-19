import pytest

from ai_review.config import settings
from ai_review.services.review.gateway.review_dry_run_comment_gateway import ReviewDryRunCommentGateway
from ai_review.services.review.internal.inline.schema import InlineCommentSchema, InlineCommentListSchema
from ai_review.services.review.internal.inline_reply.schema import InlineCommentReplySchema
from ai_review.services.review.internal.summary.schema import SummaryCommentSchema
from ai_review.services.review.internal.summary_reply.schema import SummaryCommentReplySchema
from ai_review.services.vcs.types import ReviewCommentSchema
from ai_review.tests.fixtures.services.artifacts import FakeArtifactsService
from ai_review.tests.fixtures.services.vcs import FakeVCSClient


@pytest.mark.asyncio
async def test_process_inline_reply_dry_run_logs_and_no_vcs_calls(
        capsys: pytest.CaptureFixture,
        fake_vcs_client: FakeVCSClient,
        fake_artifacts_service: FakeArtifactsService,
        review_dry_run_comment_gateway: ReviewDryRunCommentGateway
):
    reply = InlineCommentReplySchema(message="AI reply dry-run")
    await review_dry_run_comment_gateway.process_inline_reply("t1", reply)
    output = capsys.readouterr().out

    assert "[dry-run]" in output
    assert "Would create inline reply" in output
    assert not any(call[0].startswith("create_") for call in fake_vcs_client.calls)

    assert ("save_vcs_inline_reply", {"thread_id": "t1", "reply": reply}) in fake_artifacts_service.calls


@pytest.mark.asyncio
async def test_process_summary_reply_dry_run_logs_and_no_vcs_calls(
        capsys: pytest.CaptureFixture,
        fake_vcs_client: FakeVCSClient,
        fake_artifacts_service: FakeArtifactsService,
        review_dry_run_comment_gateway: ReviewDryRunCommentGateway
):
    reply = SummaryCommentReplySchema(text="Dry-run summary reply")
    await review_dry_run_comment_gateway.process_summary_reply("t2", reply)
    output = capsys.readouterr().out

    assert "[dry-run]" in output
    assert "Would create summary reply" in output
    assert not any(call[0].startswith("create_") for call in fake_vcs_client.calls)

    assert ("save_vcs_summary_reply", {"thread_id": "t2", "reply": reply}) in fake_artifacts_service.calls


@pytest.mark.asyncio
async def test_process_inline_comment_dry_run_logs_and_no_vcs_calls(
        capsys: pytest.CaptureFixture,
        fake_vcs_client: FakeVCSClient,
        fake_artifacts_service: FakeArtifactsService,
        review_dry_run_comment_gateway: ReviewDryRunCommentGateway
):
    comment = InlineCommentSchema(file="a.py", line=10, message="Test comment")
    await review_dry_run_comment_gateway.process_inline_comment(comment)
    output = capsys.readouterr().out

    assert "[dry-run]" in output
    assert "Would create inline comment" in output
    assert "a.py" in output
    assert not any(call[0].startswith("create_") for call in fake_vcs_client.calls)

    assert ("save_vcs_inline", {"comment": comment}) in fake_artifacts_service.calls


@pytest.mark.asyncio
async def test_process_summary_comment_dry_run_logs_and_no_vcs_calls(
        capsys: pytest.CaptureFixture,
        fake_vcs_client: FakeVCSClient,
        fake_artifacts_service: FakeArtifactsService,
        review_dry_run_comment_gateway: ReviewDryRunCommentGateway
):
    comment = SummaryCommentSchema(text="Dry-run summary comment")
    await review_dry_run_comment_gateway.process_summary_comment(comment)
    output = capsys.readouterr().out

    assert "[dry-run]" in output
    assert "Would create summary comment" in output
    assert not any(call[0].startswith("create_") for call in fake_vcs_client.calls)

    assert ("save_vcs_summary", {"comment": comment}) in fake_artifacts_service.calls


@pytest.mark.asyncio
async def test_process_summary_comment_dry_run_logs_update_and_delete_when_replace_enabled(
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
        fake_vcs_client: FakeVCSClient,
        fake_artifacts_service: FakeArtifactsService,
        review_dry_run_comment_gateway: ReviewDryRunCommentGateway,
):
    monkeypatch.setattr(settings.review, "summary_replace_previous", True)

    comment = SummaryCommentSchema(text="Dry-run summary comment v2")
    previous = [
        ReviewCommentSchema(id="10", body="old-1"),
        ReviewCommentSchema(id="11", body="old-2-latest"),
    ]

    await review_dry_run_comment_gateway.process_summary_comment(comment, previous=previous)
    output = capsys.readouterr().out

    assert "[dry-run] Would update summary comment 11 / delete 1 older" in output
    assert not any(call[0].startswith("create_") for call in fake_vcs_client.calls)
    assert not any(call[0].startswith("update_") for call in fake_vcs_client.calls)
    assert not any(call[0].startswith("delete_") for call in fake_vcs_client.calls)

    assert ("save_vcs_summary", {"comment": comment}) in fake_artifacts_service.calls


@pytest.mark.asyncio
async def test_process_inline_comments_iterates_all(
        capsys: pytest.CaptureFixture,
        fake_vcs_client: FakeVCSClient,
        fake_artifacts_service: FakeArtifactsService,
        review_dry_run_comment_gateway: ReviewDryRunCommentGateway
):
    comments = InlineCommentListSchema(root=[
        InlineCommentSchema(file="a.py", line=1, message="C1"),
        InlineCommentSchema(file="b.py", line=2, message="C2"),
    ])
    await review_dry_run_comment_gateway.process_inline_comments(comments)
    output = capsys.readouterr().out

    assert "[dry-run]" in output
    assert "a.py" in output
    assert "b.py" in output
    assert not any(call[0].startswith("create_") for call in fake_vcs_client.calls)

    assert ("save_vcs_inline", {"comment": comments.root[0]}) in fake_artifacts_service.calls
    assert ("save_vcs_inline", {"comment": comments.root[1]}) in fake_artifacts_service.calls


@pytest.mark.asyncio
async def test_clear_inline_comments_dry_run_no_comments(
        capsys: pytest.CaptureFixture,
        fake_vcs_client: FakeVCSClient,
        review_dry_run_comment_gateway: ReviewDryRunCommentGateway,
):
    fake_vcs_client.responses["get_inline_comments"] = []

    await review_dry_run_comment_gateway.clear_inline_comments()
    output = capsys.readouterr().out

    assert "[dry-run] No AI inline comments to clear" in output
    assert not any(call[0].startswith("delete_") for call in fake_vcs_client.calls)


@pytest.mark.asyncio
async def test_clear_inline_comments_dry_run_logs_each_comment(
        capsys: pytest.CaptureFixture,
        fake_vcs_client: FakeVCSClient,
        review_dry_run_comment_gateway: ReviewDryRunCommentGateway,
):
    fake_vcs_client.responses["get_inline_comments"] = [
        ReviewCommentSchema(id="1", body=f"AI inline\n\n{settings.review.inline_tag}"),
        ReviewCommentSchema(id="2", body=f"AI inline\n\n{settings.review.inline_tag}"),
    ]

    await review_dry_run_comment_gateway.clear_inline_comments()
    output = capsys.readouterr().out

    assert "[dry-run] Would clear 2 AI inline comments" in output
    assert "[dry-run] Would delete inline comment 1" in output
    assert "[dry-run] Would delete inline comment 2" in output

    assert not any(call[0].startswith("delete_") for call in fake_vcs_client.calls)


@pytest.mark.asyncio
async def test_clear_summary_comments_dry_run_no_comments(
        capsys: pytest.CaptureFixture,
        fake_vcs_client: FakeVCSClient,
        review_dry_run_comment_gateway: ReviewDryRunCommentGateway,
):
    fake_vcs_client.responses["get_general_comments"] = []

    await review_dry_run_comment_gateway.clear_summary_comments()
    output = capsys.readouterr().out

    assert "[dry-run] No AI summary comments to clear" in output
    assert not any(call[0].startswith("delete_") for call in fake_vcs_client.calls)


@pytest.mark.asyncio
async def test_clear_summary_comments_dry_run_logs_each_comment(
        capsys: pytest.CaptureFixture,
        fake_vcs_client: FakeVCSClient,
        review_dry_run_comment_gateway: ReviewDryRunCommentGateway,
):
    fake_vcs_client.responses["get_general_comments"] = [
        ReviewCommentSchema(id="10", body=f"First AI summary\n\n{settings.review.summary_tag}"),
        ReviewCommentSchema(id="11", body=f"Second AI summary\n\n{settings.review.summary_tag}"),
    ]

    await review_dry_run_comment_gateway.clear_summary_comments()
    output = capsys.readouterr().out

    assert "[dry-run] Would clear 2 AI summary comments" in output
    assert "[dry-run] Would delete summary comment 10" in output
    assert "[dry-run] Would delete summary comment 11" in output

    assert not any(call[0].startswith("delete_") for call in fake_vcs_client.calls)
