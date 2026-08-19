from ai_review.config import settings
from ai_review.services.review.internal.summary.schema import SummaryCommentSchema


def test_normalize_text_strips_whitespace():
    comment = SummaryCommentSchema(text="   some summary   ")
    assert comment.text == "some summary"


def test_normalize_text_empty_becomes_empty_string():
    comment = SummaryCommentSchema(text="     ")
    assert comment.text == ""


def test_body_with_tag_appends_tag(monkeypatch):
    monkeypatch.setattr(settings.review, "summary_tag", "#ai-summary")
    comment = SummaryCommentSchema(text="Review passed")
    body = comment.body_with_tag
    assert body.startswith("Review passed")
    assert body.endswith("\n\n#ai-summary")
    assert "\n\n#ai-summary" in body


def test_body_with_fallback_tag_appends_fallback_tag(monkeypatch):
    monkeypatch.setattr(settings.review, "inline_fallback_tag", "#ai-inline-fallback")
    comment = SummaryCommentSchema(text="Fallback comment")
    body = comment.body_with_fallback_tag
    assert body.startswith("Fallback comment")
    assert body.endswith("\n\n#ai-inline-fallback")
    assert "\n\n#ai-inline-fallback" in body


def test_body_with_tag_prepends_rendered_header_when_configured(monkeypatch):
    monkeypatch.setattr(settings.review, "summary_tag", "#ai-summary")
    monkeypatch.setattr(settings.review, "summary_header", "### AI review: {model}")
    monkeypatch.setattr(settings.llm.meta, "model", "gpt-4o-mini")

    comment = SummaryCommentSchema(text="Review passed")
    body = comment.body_with_tag

    assert body == "### AI review: gpt-4o-mini\n\nReview passed\n\n#ai-summary"


def test_body_with_tag_omits_header_when_not_configured(monkeypatch):
    monkeypatch.setattr(settings.review, "summary_tag", "#ai-summary")
    monkeypatch.setattr(settings.review, "summary_header", "")

    comment = SummaryCommentSchema(text="Review passed")
    body = comment.body_with_tag

    assert body == "Review passed\n\n#ai-summary"
    assert "###" not in body


def test_body_with_fallback_tag_ignores_summary_header(monkeypatch):
    monkeypatch.setattr(settings.review, "inline_fallback_tag", "#ai-inline-fallback")
    monkeypatch.setattr(settings.review, "summary_header", "### AI review: {model}")

    comment = SummaryCommentSchema(text="Fallback comment")
    body = comment.body_with_fallback_tag

    assert body == "Fallback comment\n\n#ai-inline-fallback"
