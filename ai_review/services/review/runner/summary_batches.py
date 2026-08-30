import re

from ai_review.services.diff.schema import DiffFileSchema
from ai_review.services.diff.service import COVERAGE_NOTICE_FILE, TRUNCATION_NOTICE_FILE

NOTICE_FILES = {TRUNCATION_NOTICE_FILE, COVERAGE_NOTICE_FILE}

SCORE_PATTERN = re.compile(r"overall score:\s*([0-9.]+)", re.IGNORECASE)


def real_files(batch: list[DiffFileSchema]) -> list[str]:
    return [entry.file for entry in batch if entry.file not in NOTICE_FILES]


def extract_and_strip_score(text: str) -> tuple[str, float | None]:
    score: float | None = None
    kept_lines: list[str] = []
    for line in text.splitlines():
        match = SCORE_PATTERN.search(line)
        if match and score is None:
            score = float(match.group(1))
            continue
        kept_lines.append(line)

    return "\n".join(kept_lines).strip(), score


def consolidate_batches(
        batch_texts: list[str],
        batches: list[list[DiffFileSchema]],
        total_changed_files: int,
) -> str:
    total_batches = len(batch_texts)
    reviewed_files = sum(len(real_files(batch)) for batch in batches)

    sections = [
        f"Batched review: {total_batches} parts covering {reviewed_files} of {total_changed_files} "
        f"changed files."
    ]
    scores: list[float] = []

    for i, (text, batch) in enumerate(zip(batch_texts, batches), start=1):
        stripped_text = (text or "").strip()
        if not stripped_text:
            sections.append(f"## Part {i}/{total_batches} — no result (skipped)")
            continue

        body, score = extract_and_strip_score(stripped_text)
        if score is not None:
            scores.append(score)

        n_files = len(real_files(batch))
        sections.append(f"## Part {i}/{total_batches} — {n_files} files\n\n{body}")

    if scores:
        sections.append(f"Overall score: {min(scores):.1f}/10")

    return "\n\n".join(sections)
