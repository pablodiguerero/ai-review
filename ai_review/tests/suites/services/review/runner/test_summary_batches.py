from ai_review.services.diff.schema import DiffFileSchema
from ai_review.services.review.runner.summary_batches import (
    consolidate_batches,
    extract_and_strip_score,
    real_files,
)


def test_real_files_filters_out_notice_entries() -> None:
    batch = [
        DiffFileSchema(file="a.py", diff="diff a"),
        DiffFileSchema(file="ai-review: coverage notice", diff="notice"),
        DiffFileSchema(file="ai-review: truncation notice", diff="notice"),
        DiffFileSchema(file="b.py", diff="diff b"),
    ]
    assert real_files(batch) == ["a.py", "b.py"]


def test_extract_and_strip_score_removes_line_and_returns_value() -> None:
    text = "Some findings here.\n\nOverall score: 8.5\n\nMore notes."
    body, score = extract_and_strip_score(text)

    assert score == 8.5
    assert "Overall score" not in body
    assert "Some findings here." in body
    assert "More notes." in body


def test_extract_and_strip_score_returns_none_when_absent() -> None:
    body, score = extract_and_strip_score("Just findings, no score line.")

    assert score is None
    assert body == "Just findings, no score line."


def test_consolidate_batches_builds_sections_strips_scores_and_uses_min() -> None:
    batch_texts = [
        "Findings for a.\n\nOverall score: 9",
        "Findings for b.\n\nOverall score: 6",
    ]
    batches = [
        [DiffFileSchema(file="a.py", diff="diff a")],
        [DiffFileSchema(file="b.py", diff="diff b")],
    ]

    result = consolidate_batches(batch_texts, batches, total_changed_files=2)

    assert "Batched review: 2 parts covering 2 of 2 changed files." in result
    assert "## Part 1/2 — 1 files" in result
    assert "## Part 2/2 — 1 files" in result
    assert "Findings for a." in result
    assert "Findings for b." in result
    assert "Overall score: 9" not in result
    assert result.strip().endswith("Overall score: 6.0/10")


def test_consolidate_batches_marks_empty_batch_as_skipped_and_excludes_from_scores() -> None:
    batch_texts = ["", "Findings for b.\n\nOverall score: 6"]
    batches = [
        [DiffFileSchema(file="a.py", diff="diff a")],
        [DiffFileSchema(file="b.py", diff="diff b")],
    ]

    result = consolidate_batches(batch_texts, batches, total_changed_files=2)

    assert "## Part 1/2 — no result (skipped)" in result
    assert "## Part 2/2 — 1 files" in result
    assert result.strip().endswith("Overall score: 6.0/10")


def test_consolidate_batches_omits_score_line_when_no_part_has_a_score() -> None:
    batch_texts = ["Findings for a, no score.", "Findings for b, no score."]
    batches = [
        [DiffFileSchema(file="a.py", diff="diff a")],
        [DiffFileSchema(file="b.py", diff="diff b")],
    ]

    result = consolidate_batches(batch_texts, batches, total_changed_files=2)

    assert "Overall score:" not in result


def test_consolidate_batches_ignores_notice_entries_when_counting_files() -> None:
    batch_texts = ["Findings.\n\nOverall score: 7"]
    batches = [
        [
            DiffFileSchema(file="a.py", diff="diff a"),
            DiffFileSchema(file="ai-review: coverage notice", diff="notice"),
        ],
    ]

    result = consolidate_batches(batch_texts, batches, total_changed_files=1)

    assert "## Part 1/1 — 1 files" in result
