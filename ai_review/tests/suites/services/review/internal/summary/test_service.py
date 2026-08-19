import pytest

from ai_review.config import settings
from ai_review.services.review.internal.summary.schema import SummaryCommentSchema
from ai_review.services.review.internal.summary.service import SummaryCommentService


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Some summary", "Some summary"),
        ("   padded summary   ", "padded summary"),
        ("", ""),
        (None, ""),
    ]
)
def test_parse_model_output_normalizes_and_wraps(raw: str | None, expected: str):
    result = SummaryCommentService.parse_model_output(raw)
    assert isinstance(result, SummaryCommentSchema)
    assert result.text == expected


def test_parse_model_output_normalizes_tables_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.review, "summary_normalize_tables", True)
    raw = "Intro\n\nCriterion | Rating\nCorrectness | 5\n\nOutro"

    result = SummaryCommentService.parse_model_output(raw)

    assert "| Criterion | Rating |" in result.text
    assert "| --- | --- |" in result.text
    assert "| Correctness | 5 |" in result.text


def test_parse_model_output_leaves_tables_untouched_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.review, "summary_normalize_tables", False)
    raw = "Intro\n\nCriterion | Rating\nCorrectness | 5\n\nOutro"

    result = SummaryCommentService.parse_model_output(raw)

    assert result.text == raw
