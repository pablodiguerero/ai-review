from enum import StrEnum


class ReviewOutcome(StrEnum):
    POSTED = "POSTED"
    SKIPPED = "SKIPPED"
    EMPTY = "EMPTY"
