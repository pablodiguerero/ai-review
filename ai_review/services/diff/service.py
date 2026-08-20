from ai_review.config import settings
from ai_review.libs.config.review import ReviewMode
from ai_review.libs.diff.models import Diff
from ai_review.libs.diff.parser import DiffParser
from ai_review.libs.logger import get_logger
from ai_review.services.diff.renderers import (
    build_full_file_diff,
    build_full_file_current,
    build_full_file_previous,
    build_only_added,
    build_only_removed,
    build_added_and_removed,
    build_only_added_with_context,
    build_only_removed_with_context,
    build_added_and_removed_with_context
)
from ai_review.services.diff.schema import DiffFileSchema
from ai_review.services.diff.tools import find_diff_file
from ai_review.services.diff.types import DiffServiceProtocol
from ai_review.services.git.types import GitServiceProtocol

logger = get_logger("DIFF_SERVICE")

TRUNCATION_MARKER = "\n... file diff truncated ..."
TRUNCATION_NOTICE_FILE = "ai-review: truncation notice"
COVERAGE_NOTICE_FILE = "ai-review: coverage notice"


class DiffService(DiffServiceProtocol):
    @classmethod
    def parse(cls, raw_diff: str) -> Diff:
        if not raw_diff.strip():
            logger.debug("Received empty diff string")
            return Diff(files=[], raw=raw_diff)

        try:
            return DiffParser.parse(raw_diff)
        except Exception as error:
            logger.exception(f"Failed to parse diff: {error}")
            raise

    @classmethod
    def render_file(
            cls,
            file: str,
            raw_diff: str,
            base_sha: str | None = None,
            head_sha: str | None = None,
    ) -> DiffFileSchema:
        diff = cls.parse(raw_diff)
        target = find_diff_file(diff, file)

        match settings.review.mode:
            case ReviewMode.FULL_FILE_CURRENT:
                file_diff = build_full_file_current(target, file, head_sha)
            case ReviewMode.FULL_FILE_PREVIOUS:
                file_diff = build_full_file_previous(target, file, base_sha)
            case ReviewMode.FULL_FILE_DIFF:
                file_diff = build_full_file_diff(target)
            case ReviewMode.ONLY_ADDED:
                file_diff = build_only_added(target)
            case ReviewMode.ONLY_REMOVED:
                file_diff = build_only_removed(target)
            case ReviewMode.ADDED_AND_REMOVED:
                file_diff = build_added_and_removed(target)
            case ReviewMode.ONLY_ADDED_WITH_CONTEXT:
                file_diff = build_only_added_with_context(target, settings.review.context_lines)
            case ReviewMode.ONLY_REMOVED_WITH_CONTEXT:
                file_diff = build_only_removed_with_context(target, settings.review.context_lines)
            case ReviewMode.ADDED_AND_REMOVED_WITH_CONTEXT:
                file_diff = build_added_and_removed_with_context(target, settings.review.context_lines)
            case _:
                file_diff = f"# Unsupported mode: {settings.review.mode}"

        return DiffFileSchema(diff=file_diff, file=file)

    @classmethod
    def render_files(
            cls,
            git: GitServiceProtocol,
            files: list[str],
            base_sha: str,
            head_sha: str,
    ) -> list[DiffFileSchema]:
        return cls._cap_by_max_diff_chars(cls._render_all(git, files, base_sha, head_sha))

    @classmethod
    def render_batches(
            cls,
            git: GitServiceProtocol,
            files: list[str],
            base_sha: str,
            head_sha: str,
    ) -> list[list[DiffFileSchema]]:
        return cls._group_into_batches(cls._render_all(git, files, base_sha, head_sha))

    @classmethod
    def _render_all(
            cls,
            git: GitServiceProtocol,
            files: list[str],
            base_sha: str,
            head_sha: str,
    ) -> list[DiffFileSchema]:
        annotated: list[DiffFileSchema] = []
        for file in files:
            raw_diff = git.get_diff_for_file(base_sha, head_sha, file)
            if not raw_diff.strip():
                logger.debug(f"No diff for {file}, skipping")
                continue

            annotated.append(
                cls.render_file(
                    file=file,
                    base_sha=base_sha,
                    head_sha=head_sha,
                    raw_diff=raw_diff,
                )
            )

        return annotated

    @classmethod
    def _truncate_entry(cls, entry: DiffFileSchema, budget: int) -> DiffFileSchema:
        cut = max(budget - len(TRUNCATION_MARKER), 0)
        return DiffFileSchema(file=entry.file, diff=entry.diff[:cut] + TRUNCATION_MARKER)

    @classmethod
    def _cap_by_max_diff_chars(cls, entries: list[DiffFileSchema]) -> list[DiffFileSchema]:
        budget = settings.review.max_diff_chars
        if budget is None or not entries:
            return entries

        total_chars = sum(len(entry.diff) for entry in entries)
        if total_chars <= budget:
            return entries

        kept: list[DiffFileSchema] = []
        running = 0
        for entry in entries:
            entry_len = len(entry.diff)

            if not kept and entry_len > budget:
                truncated = cls._truncate_entry(entry, budget)
                kept.append(truncated)
                running += len(truncated.diff)
                continue

            if kept and running + entry_len > budget:
                break

            kept.append(entry)
            running += entry_len

        kept_chars = sum(len(entry.diff) for entry in kept)
        omitted_files = len(entries) - len(kept)
        omitted_chars = total_chars - kept_chars

        logger.warning(
            f"Diff exceeds REVIEW__MAX_DIFF_CHARS={budget}: keeping {len(kept)}/{len(entries)} files "
            f"({omitted_files} omitted, ~{omitted_chars} characters not shown)"
        )

        notice = (
            f"Only {len(kept)} of {len(entries)} changed files are shown below "
            f"({omitted_files} files omitted, ~{omitted_chars} characters) because the full diff exceeds "
            f"REVIEW__MAX_DIFF_CHARS={budget}. This is a partial review of the largest/first files; "
            f"unshown files were not analyzed."
        )
        kept.append(DiffFileSchema(file=TRUNCATION_NOTICE_FILE, diff=notice))
        return kept

    @classmethod
    def _group_into_batches(cls, entries: list[DiffFileSchema]) -> list[list[DiffFileSchema]]:
        if not entries:
            return []

        budget = settings.review.max_diff_chars
        if budget is None:
            return [entries]

        batches: list[list[DiffFileSchema]] = []
        current: list[DiffFileSchema] = []
        current_chars = 0

        for entry in entries:
            entry_len = len(entry.diff)

            if entry_len > budget:
                if current:
                    batches.append(current)
                    current = []
                    current_chars = 0
                batches.append([cls._truncate_entry(entry, budget)])
                continue

            if current and current_chars + entry_len > budget:
                batches.append(current)
                current = [entry]
                current_chars = entry_len
                continue

            current.append(entry)
            current_chars += entry_len

        if current:
            batches.append(current)

        max_batches = settings.review.max_diff_batches
        if len(batches) <= max_batches:
            return batches

        kept_batches = batches[:max_batches]
        dropped_batches = batches[max_batches:]
        dropped_files = sum(len(batch) for batch in dropped_batches)
        dropped_chars = sum(len(entry.diff) for batch in dropped_batches for entry in batch)
        total_files = len(entries)

        logger.warning(
            f"Diff exceeds REVIEW__MAX_DIFF_BATCHES={max_batches}: dropping {dropped_files}/{total_files} "
            f"files (~{dropped_chars} characters not reviewed)"
        )

        notice = (
            f"{dropped_files} of {total_files} files not reviewed: exceeds "
            f"REVIEW__MAX_DIFF_BATCHES={max_batches} batches of REVIEW__MAX_DIFF_CHARS={budget}"
        )
        kept_batches[-1] = kept_batches[-1] + [DiffFileSchema(file=COVERAGE_NOTICE_FILE, diff=notice)]
        return kept_batches
