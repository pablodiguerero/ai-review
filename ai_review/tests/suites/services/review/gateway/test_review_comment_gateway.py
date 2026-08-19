import pytest

from ai_review.config import settings
from ai_review.services.review.gateway.review_comment_gateway import ReviewCommentGateway, body_has_tag
from ai_review.services.review.internal.inline.schema import InlineCommentSchema, InlineCommentListSchema
from ai_review.services.review.internal.inline_reply.schema import InlineCommentReplySchema
from ai_review.services.review.internal.summary.schema import SummaryCommentSchema
from ai_review.services.review.internal.summary_reply.schema import SummaryCommentReplySchema
from ai_review.services.vcs.types import ReviewThreadSchema, ReviewCommentSchema, ThreadKind
from ai_review.tests.fixtures.services.artifacts import FakeArtifactsService
from ai_review.tests.fixtures.services.vcs import FakeVCSClient


@pytest.mark.asyncio
async def test_get_inline_threads_filters_by_tag(
        fake_vcs_client: FakeVCSClient,
        review_comment_gateway: ReviewCommentGateway,
):
    threads = [
        ReviewThreadSchema(
            id="1",
            kind=ThreadKind.INLINE,
            file="a.py",
            comments=[ReviewCommentSchema(id="1", body=f"Hello\n\n{settings.review.inline_reply_tag}")]
        ),
        ReviewThreadSchema(
            id="2",
            kind=ThreadKind.INLINE,
            file="b.py",
            comments=[ReviewCommentSchema(id="2", body="No AI tag here")]
        ),
    ]
    fake_vcs_client.responses["get_inline_threads"] = threads

    result = await review_comment_gateway.get_inline_threads()

    assert len(result) == 1
    assert result[0].id == "1"
    assert any(call[0] == "get_inline_threads" for call in fake_vcs_client.calls)


@pytest.mark.asyncio
async def test_get_summary_threads_filters_by_tag(
        fake_vcs_client: FakeVCSClient,
        review_comment_gateway: ReviewCommentGateway,
):
    threads = [
        ReviewThreadSchema(
            id="10",
            kind=ThreadKind.SUMMARY,
            comments=[ReviewCommentSchema(id="1", body=f"AI\n\n{settings.review.summary_reply_tag}")]
        ),
        ReviewThreadSchema(
            id="11",
            kind=ThreadKind.SUMMARY,
            comments=[ReviewCommentSchema(id="2", body="No tags here")]
        ),
    ]
    fake_vcs_client.responses["get_general_threads"] = threads

    result = await review_comment_gateway.get_summary_threads()

    assert len(result) == 1
    assert result[0].id == "10"
    assert any(call[0] == "get_general_threads" for call in fake_vcs_client.calls)


@pytest.mark.asyncio
async def test_get_inline_comments_filters_only_ai_comments(
        fake_vcs_client: FakeVCSClient,
        review_comment_gateway: ReviewCommentGateway,
):
    fake_vcs_client.responses["get_inline_comments"] = [
        ReviewCommentSchema(id="1", body=f"AI comment\n\n{settings.review.inline_tag}"),
        ReviewCommentSchema(id="2", body="Regular inline comment"),
    ]

    result = await review_comment_gateway.get_inline_comments()

    assert len(result) == 1
    assert result[0].id == "1"

    assert any(call[0] == "get_inline_comments" for call in fake_vcs_client.calls)


@pytest.mark.asyncio
async def test_get_inline_comments_returns_empty_when_no_ai_comments(
        fake_vcs_client: FakeVCSClient,
        review_comment_gateway: ReviewCommentGateway,
):
    fake_vcs_client.responses["get_inline_comments"] = [
        ReviewCommentSchema(id="1", body="Just a comment"),
    ]

    result = await review_comment_gateway.get_inline_comments()

    assert result == []


@pytest.mark.asyncio
async def test_get_summary_comments_filters_only_ai_comments(
        fake_vcs_client: FakeVCSClient,
        review_comment_gateway: ReviewCommentGateway,
):
    fake_vcs_client.responses["get_general_comments"] = [
        ReviewCommentSchema(id="10", body=f"AI summary\n\n{settings.review.summary_tag}"),
        ReviewCommentSchema(id="11", body="Regular summary"),
    ]

    result = await review_comment_gateway.get_summary_comments()

    assert len(result) == 1
    assert result[0].id == "10"

    assert any(call[0] == "get_general_comments" for call in fake_vcs_client.calls)


@pytest.mark.asyncio
async def test_get_summary_comments_returns_empty_when_no_ai_comments(
        fake_vcs_client: FakeVCSClient,
        review_comment_gateway: ReviewCommentGateway,
):
    fake_vcs_client.responses["get_general_comments"] = [
        ReviewCommentSchema(id="1", body="Regular comment"),
    ]

    result = await review_comment_gateway.get_summary_comments()

    assert result == []


@pytest.mark.asyncio
async def test_process_inline_reply_happy_path(
        fake_vcs_client: FakeVCSClient,
        fake_artifacts_service: FakeArtifactsService,
        review_comment_gateway: ReviewCommentGateway,
):
    reply = InlineCommentReplySchema(message="AI reply text")

    await review_comment_gateway.process_inline_reply("t1", reply)

    assert any(call[0] == "create_inline_reply" for call in fake_vcs_client.calls)

    assert ("save_vcs_inline_reply", {"thread_id": "t1", "reply": reply}) in fake_artifacts_service.calls


@pytest.mark.asyncio
async def test_process_inline_reply_error(
        capsys: pytest.CaptureFixture,
        fake_vcs_client: FakeVCSClient,
        fake_artifacts_service: FakeArtifactsService,
        review_comment_gateway: ReviewCommentGateway,
):

    async def failing_create_inline_reply(thread_id: str, body: str):
        raise RuntimeError("API error")

    fake_vcs_client.create_inline_reply = failing_create_inline_reply

    reply = InlineCommentReplySchema(message="AI reply text")
    await review_comment_gateway.process_inline_reply("t1", reply)
    output = capsys.readouterr().out

    assert "Failed to create inline reply" in output

    assert all(call[0] != "save_vcs_inline_reply" for call in fake_artifacts_service.calls)


@pytest.mark.asyncio
async def test_process_summary_reply_success(
        fake_vcs_client: FakeVCSClient,
        fake_artifacts_service: FakeArtifactsService,
        review_comment_gateway: ReviewCommentGateway,
):
    reply = SummaryCommentReplySchema(text="AI summary reply")
    await review_comment_gateway.process_summary_reply("t42", reply)
    assert any(call[0] == "create_summary_reply" for call in fake_vcs_client.calls)

    assert ("save_vcs_summary_reply", {"thread_id": "t42", "reply": reply}) in fake_artifacts_service.calls


@pytest.mark.asyncio
async def test_process_summary_reply_error(
        capsys: pytest.CaptureFixture,
        fake_vcs_client: FakeVCSClient,
        review_comment_gateway: ReviewCommentGateway,
):

    async def failing_create_summary_reply(thread_id: str, body: str):
        raise RuntimeError("Network fail")

    fake_vcs_client.create_summary_reply = failing_create_summary_reply

    reply = SummaryCommentReplySchema(text="AI summary reply")
    await review_comment_gateway.process_summary_reply("t42", reply)
    output = capsys.readouterr().out

    assert "Failed to create summary reply" in output


@pytest.mark.asyncio
async def test_process_inline_comment_happy_path(
        fake_vcs_client: FakeVCSClient,
        fake_artifacts_service: FakeArtifactsService,
        review_comment_gateway: ReviewCommentGateway,
):
    comment = InlineCommentSchema(file="f.py", line=1, message="AI inline comment")
    await review_comment_gateway.process_inline_comment(comment)
    assert any(call[0] == "create_inline_comment" for call in fake_vcs_client.calls)

    assert ("save_vcs_inline", {"comment": comment}) in fake_artifacts_service.calls
    assert all(call[0] != "save_vcs_summary" for call in fake_artifacts_service.calls)
    assert all(call[0] != "save_vcs_summary_reply" for call in fake_artifacts_service.calls)


@pytest.mark.asyncio
async def test_process_inline_comment_error_fallback(
        capsys: pytest.CaptureFixture,
        fake_vcs_client: FakeVCSClient,
        fake_artifacts_service: FakeArtifactsService,
        review_comment_gateway: ReviewCommentGateway,
):

    async def failing_create_inline_comment(file: str, line: int, message: str):
        raise RuntimeError("Failed to post inline")

    fake_vcs_client.create_inline_comment = failing_create_inline_comment

    comment = InlineCommentSchema(file="x.py", line=5, message="AI inline")
    await review_comment_gateway.process_inline_comment(comment)
    output = capsys.readouterr().out

    assert "Falling back to general comment" in output
    assert any(call[0] == "create_general_comment" for call in fake_vcs_client.calls)

    fallback_call = next(call for call in fake_vcs_client.calls if call[0] == "create_general_comment")
    posted_body = fallback_call[1][0]
    assert settings.review.inline_fallback_tag in posted_body
    assert settings.review.summary_tag not in posted_body

    assert all(call[0] != "save_vcs_inline" for call in fake_artifacts_service.calls)
    assert any(call[0] == "save_vcs_summary" for call in fake_artifacts_service.calls)


@pytest.mark.asyncio
async def test_process_summary_comment_happy_path(
        fake_vcs_client: FakeVCSClient,
        fake_artifacts_service: FakeArtifactsService,
        review_comment_gateway: ReviewCommentGateway,
):
    comment = SummaryCommentSchema(text="AI summary")
    await review_comment_gateway.process_summary_comment(comment)
    assert any(call[0] == "create_general_comment" for call in fake_vcs_client.calls)

    assert ("save_vcs_summary", {"comment": comment}) in fake_artifacts_service.calls


@pytest.mark.asyncio
async def test_process_summary_comment_error(
        capsys: pytest.CaptureFixture,
        fake_vcs_client: FakeVCSClient,
        fake_artifacts_service: FakeArtifactsService,
        review_comment_gateway: ReviewCommentGateway,
):

    async def failing_create_general_comment(body: str):
        raise RuntimeError("Backend down")

    fake_vcs_client.create_general_comment = failing_create_general_comment

    comment = SummaryCommentSchema(text="Broken")
    await review_comment_gateway.process_summary_comment(comment)
    output = capsys.readouterr().out

    assert "Failed to process summary comment" in output

    assert all(call[0] != "save_vcs_summary" for call in fake_artifacts_service.calls)


@pytest.mark.asyncio
async def test_process_summary_comment_ignores_previous_when_replace_disabled(
        monkeypatch: pytest.MonkeyPatch,
        fake_vcs_client: FakeVCSClient,
        fake_artifacts_service: FakeArtifactsService,
        review_comment_gateway: ReviewCommentGateway,
):
    monkeypatch.setattr(settings.review, "summary_replace_previous", False)

    comment = SummaryCommentSchema(text="AI summary")
    previous = [ReviewCommentSchema(id="1", body="old")]

    await review_comment_gateway.process_summary_comment(comment, previous=previous)

    assert any(call[0] == "create_general_comment" for call in fake_vcs_client.calls)
    assert all(call[0] != "update_general_comment" for call in fake_vcs_client.calls)
    assert all(call[0] != "delete_general_comment" for call in fake_vcs_client.calls)


@pytest.mark.asyncio
async def test_process_summary_comment_replaces_latest_and_deletes_older(
        monkeypatch: pytest.MonkeyPatch,
        fake_vcs_client: FakeVCSClient,
        fake_artifacts_service: FakeArtifactsService,
        review_comment_gateway: ReviewCommentGateway,
):
    monkeypatch.setattr(settings.review, "summary_replace_previous", True)

    comment = SummaryCommentSchema(text="AI summary v2")
    previous = [
        ReviewCommentSchema(id="1", body="old-1"),
        ReviewCommentSchema(id="2", body="old-2"),
        ReviewCommentSchema(id="3", body="old-3-latest"),
    ]

    await review_comment_gateway.process_summary_comment(comment, previous=previous)

    assert all(call[0] != "create_general_comment" for call in fake_vcs_client.calls)

    update_calls = [call for call in fake_vcs_client.calls if call[0] == "update_general_comment"]
    assert len(update_calls) == 1
    assert update_calls[0][1][0] == "3"

    delete_calls = [call for call in fake_vcs_client.calls if call[0] == "delete_general_comment"]
    assert {call[1][0] for call in delete_calls} == {"1", "2"}

    assert ("save_vcs_summary", {"comment": comment}) in fake_artifacts_service.calls


@pytest.mark.asyncio
async def test_process_summary_comment_replace_with_single_previous_deletes_nothing(
        monkeypatch: pytest.MonkeyPatch,
        fake_vcs_client: FakeVCSClient,
        review_comment_gateway: ReviewCommentGateway,
):
    monkeypatch.setattr(settings.review, "summary_replace_previous", True)

    comment = SummaryCommentSchema(text="AI summary v2")
    previous = [ReviewCommentSchema(id="1", body="old-1")]

    await review_comment_gateway.process_summary_comment(comment, previous=previous)

    update_calls = [call for call in fake_vcs_client.calls if call[0] == "update_general_comment"]
    assert len(update_calls) == 1
    assert update_calls[0][1][0] == "1"
    assert all(call[0] != "delete_general_comment" for call in fake_vcs_client.calls)


@pytest.mark.asyncio
async def test_process_summary_comment_falls_back_to_create_when_update_fails(
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
        fake_vcs_client: FakeVCSClient,
        fake_artifacts_service: FakeArtifactsService,
        review_comment_gateway: ReviewCommentGateway,
):
    monkeypatch.setattr(settings.review, "summary_replace_previous", True)

    async def failing_update_general_comment(comment_id, message):
        raise RuntimeError("Update not supported")

    fake_vcs_client.update_general_comment = failing_update_general_comment

    comment = SummaryCommentSchema(text="AI summary v2")
    previous = [ReviewCommentSchema(id="1", body="old-1")]

    await review_comment_gateway.process_summary_comment(comment, previous=previous)
    output = capsys.readouterr().out

    assert "falling back to create" in output.lower()
    assert any(call[0] == "create_general_comment" for call in fake_vcs_client.calls)
    assert ("save_vcs_summary", {"comment": comment}) in fake_artifacts_service.calls


@pytest.mark.asyncio
async def test_process_summary_comment_replace_enabled_but_no_previous_creates(
        monkeypatch: pytest.MonkeyPatch,
        fake_vcs_client: FakeVCSClient,
        review_comment_gateway: ReviewCommentGateway,
):
    monkeypatch.setattr(settings.review, "summary_replace_previous", True)

    comment = SummaryCommentSchema(text="AI summary")
    await review_comment_gateway.process_summary_comment(comment, previous=None)

    assert any(call[0] == "create_general_comment" for call in fake_vcs_client.calls)
    assert all(call[0] != "update_general_comment" for call in fake_vcs_client.calls)


@pytest.mark.asyncio
async def test_process_inline_fallback_comment_happy_path(
        fake_vcs_client: FakeVCSClient,
        fake_artifacts_service: FakeArtifactsService,
        review_comment_gateway: ReviewCommentGateway,
):
    comment = SummaryCommentSchema(text="**x.py:42** — missing check")
    await review_comment_gateway.process_inline_fallback_comment(comment)

    assert any(call[0] == "create_general_comment" for call in fake_vcs_client.calls)

    fallback_call = next(call for call in fake_vcs_client.calls if call[0] == "create_general_comment")
    posted_body = fallback_call[1][0]
    assert settings.review.inline_fallback_tag in posted_body
    assert settings.review.summary_tag not in posted_body

    assert ("save_vcs_summary", {"comment": comment}) in fake_artifacts_service.calls


@pytest.mark.asyncio
async def test_process_inline_fallback_comment_error(
        capsys: pytest.CaptureFixture,
        fake_vcs_client: FakeVCSClient,
        fake_artifacts_service: FakeArtifactsService,
        review_comment_gateway: ReviewCommentGateway,
):

    async def failing_create_general_comment(body: str):
        raise RuntimeError("Backend down")

    fake_vcs_client.create_general_comment = failing_create_general_comment

    comment = SummaryCommentSchema(text="Broken fallback")
    await review_comment_gateway.process_inline_fallback_comment(comment)
    output = capsys.readouterr().out

    assert "Failed to process inline fallback comment" in output

    assert all(call[0] != "save_vcs_summary" for call in fake_artifacts_service.calls)


@pytest.mark.asyncio
async def test_process_inline_comments_calls_each(
        fake_vcs_client: FakeVCSClient,
        review_comment_gateway: ReviewCommentGateway,
):
    comments = InlineCommentListSchema(root=[
        InlineCommentSchema(file="a.py", line=1, message="c1"),
        InlineCommentSchema(file="b.py", line=2, message="c2"),
    ])

    await review_comment_gateway.process_inline_comments(comments)

    created = [call for call in fake_vcs_client.calls if call[0] == "create_inline_comment"]
    assert len(created) == 2


@pytest.mark.asyncio
async def test_process_inline_comment_error_no_fallback_when_disabled(
        capsys: pytest.CaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
        fake_vcs_client: FakeVCSClient,
        fake_artifacts_service: FakeArtifactsService,
        review_comment_gateway: ReviewCommentGateway,
):
    monkeypatch.setattr(settings.review, "inline_comment_fallback", False)

    async def failing_create_inline_comment(file: str, line: int, message: str):
        raise RuntimeError("Failed to post inline")

    fake_vcs_client.create_inline_comment = failing_create_inline_comment

    comment = InlineCommentSchema(file="x.py", line=10, message="AI inline")
    await review_comment_gateway.process_inline_comment(comment)
    output = capsys.readouterr().out

    assert "Failed to process inline comment" in output
    assert "Falling back to general comment" not in output

    assert all(call[0] != "create_general_comment" for call in fake_vcs_client.calls)
    assert all(call[0] != "save_vcs_summary" for call in fake_artifacts_service.calls)
    assert all(call[0] != "save_vcs_inline" for call in fake_artifacts_service.calls)


@pytest.mark.asyncio
async def test_clear_inline_comments_deletes_all_ai_comments(
        fake_vcs_client: FakeVCSClient,
        review_comment_gateway: ReviewCommentGateway,
):
    fake_vcs_client.responses["get_inline_comments"] = [
        ReviewCommentSchema(id="1", body=f"comment 1\n\n{settings.review.inline_tag}"),
        ReviewCommentSchema(id="2", body=f"comment 2\n\n{settings.review.inline_tag}"),
    ]

    await review_comment_gateway.clear_inline_comments()

    deleted = [call for call in fake_vcs_client.calls if call[0] == "delete_inline_comment"]
    assert len(deleted) == 2
    assert {call[1][0] for call in deleted} == {"1", "2"}


@pytest.mark.asyncio
async def test_clear_inline_comments_noop_when_no_comments(
        fake_vcs_client: FakeVCSClient,
        review_comment_gateway: ReviewCommentGateway,
):
    fake_vcs_client.responses["get_inline_comments"] = []

    await review_comment_gateway.clear_inline_comments()

    assert all(call[0] != "delete_inline_comment" for call in fake_vcs_client.calls)


@pytest.mark.asyncio
async def test_clear_summary_comments_deletes_all_ai_comments(
        fake_vcs_client: FakeVCSClient,
        review_comment_gateway: ReviewCommentGateway,
):
    fake_vcs_client.responses["get_general_comments"] = [
        ReviewCommentSchema(id="10", body=f"summary 1\n\n{settings.review.summary_tag}"),
        ReviewCommentSchema(id="11", body=f"summary 2\n\n{settings.review.summary_tag}"),
    ]

    await review_comment_gateway.clear_summary_comments()

    deleted = [call for call in fake_vcs_client.calls if call[0] == "delete_general_comment"]
    assert len(deleted) == 2
    assert {call[1][0] for call in deleted} == {"10", "11"}


@pytest.mark.asyncio
async def test_clear_summary_comments_noop_when_no_comments(
        fake_vcs_client: FakeVCSClient,
        review_comment_gateway: ReviewCommentGateway,
):
    fake_vcs_client.responses["get_general_comments"] = []

    await review_comment_gateway.clear_summary_comments()

    assert all(call[0] != "delete_general_comment" for call in fake_vcs_client.calls)


@pytest.mark.asyncio
async def test_get_summary_comments_excludes_fallback_comments(
        fake_vcs_client: FakeVCSClient,
        review_comment_gateway: ReviewCommentGateway,
):
    fake_vcs_client.responses["get_general_comments"] = [
        ReviewCommentSchema(id="10", body=f"Summary\n\n{settings.review.summary_tag}"),
        ReviewCommentSchema(id="11", body=f"Fallback\n\n{settings.review.inline_fallback_tag}"),
    ]

    result = await review_comment_gateway.get_summary_comments()

    assert len(result) == 1
    assert result[0].id == "10"


def test_body_has_tag_ignores_tag_mentioned_inside_prose() -> None:
    body = "This looks similar to the #ai-review-summary format but isn't one."
    assert not body_has_tag(body, "#ai-review-summary")


def test_body_has_tag_matches_tag_on_its_own_trailing_line() -> None:
    body = "Some review text.\n\n#ai-review-summary"
    assert body_has_tag(body, "#ai-review-summary")


def test_body_has_tag_matches_tag_line_with_surrounding_whitespace() -> None:
    body = "Some review text.\n\n   #ai-review-summary   "
    assert body_has_tag(body, "#ai-review-summary")


def test_body_has_tag_does_not_cross_match_similarly_named_tags() -> None:
    grok_body = "Some review text.\n\n#ai-review-grok-summary"
    assert not body_has_tag(grok_body, "#ai-review-summary")
    assert body_has_tag(grok_body, "#ai-review-grok-summary")

    deepseek_body = "Some review text.\n\n#ai-review-summary"
    assert not body_has_tag(deepseek_body, "#ai-review-grok-summary")
    assert body_has_tag(deepseek_body, "#ai-review-summary")


@pytest.mark.asyncio
async def test_get_summary_comments_ignores_grok_comment_mentioning_deepseek_tag(
        monkeypatch: pytest.MonkeyPatch,
        fake_vcs_client: FakeVCSClient,
        review_comment_gateway: ReviewCommentGateway,
):
    monkeypatch.setattr(settings.review, "summary_tag", "#ai-review-summary")
    fake_vcs_client.responses["get_general_comments"] = [
        ReviewCommentSchema(
            id="1",
            body="A grok review that references #ai-review-summary in prose.\n\n#ai-review-grok-summary",
        ),
        ReviewCommentSchema(id="2", body=f"A deepseek review.\n\n{settings.review.summary_tag}"),
    ]

    result = await review_comment_gateway.get_summary_comments()

    assert len(result) == 1
    assert result[0].id == "2"


@pytest.mark.asyncio
async def test_process_summary_comment_replace_does_not_delete_comment_merely_mentioning_tag(
        monkeypatch: pytest.MonkeyPatch,
        fake_vcs_client: FakeVCSClient,
        fake_artifacts_service: FakeArtifactsService,
        review_comment_gateway: ReviewCommentGateway,
):
    monkeypatch.setattr(settings.review, "summary_replace_previous", True)
    monkeypatch.setattr(settings.review, "summary_tag", "#ai-review-summary")

    fake_vcs_client.responses["get_general_comments"] = [
        ReviewCommentSchema(
            id="1",
            body="A grok review that references #ai-review-summary in prose.\n\n#ai-review-grok-summary",
        ),
        ReviewCommentSchema(id="2", body=f"A deepseek review.\n\n{settings.review.summary_tag}"),
    ]

    previous = await review_comment_gateway.get_summary_comments()
    assert [c.id for c in previous] == ["2"]

    comment = SummaryCommentSchema(text="Updated deepseek review")
    await review_comment_gateway.process_summary_comment(comment, previous=previous)

    update_calls = [call for call in fake_vcs_client.calls if call[0] == "update_general_comment"]
    assert len(update_calls) == 1
    assert update_calls[0][1][0] == "2"

    assert all(call[0] != "delete_general_comment" for call in fake_vcs_client.calls)
