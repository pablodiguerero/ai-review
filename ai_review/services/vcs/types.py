from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, Field


class ThreadKind(StrEnum):
    INLINE = "INLINE"
    SUMMARY = "SUMMARY"


class UserSchema(BaseModel):
    id: str | int | None = None
    name: str = ""
    username: str = ""


class BranchRefSchema(BaseModel):
    ref: str = ""
    sha: str = ""


class ReviewInfoSchema(BaseModel):
    id: str | int | None = None
    title: str = ""
    description: str = ""
    author: UserSchema = Field(default_factory=UserSchema)
    labels: list[str] = Field(default_factory=list)
    assignees: list[UserSchema] = Field(default_factory=list)
    reviewers: list[UserSchema] = Field(default_factory=list)
    source_branch: BranchRefSchema = Field(default_factory=BranchRefSchema)
    target_branch: BranchRefSchema = Field(default_factory=BranchRefSchema)
    changed_files: list[str] = Field(default_factory=list)
    base_sha: str = ""
    head_sha: str = ""
    start_sha: str = ""


class ReviewCommentSchema(BaseModel):
    id: str | int
    body: str
    file: str | None = None
    line: int | None = None
    author: UserSchema = Field(default_factory=UserSchema)
    parent_id: str | int | None = None
    thread_id: str | int | None = None


class ReviewThreadSchema(BaseModel):
    id: str | int
    kind: ThreadKind
    file: str | None = None
    line: int | None = None
    comments: list[ReviewCommentSchema]


class VCSClientProtocol(Protocol):
    async def get_review_info(self) -> ReviewInfoSchema:
        ...

    async def get_general_comments(self) -> list[ReviewCommentSchema]:
        ...

    async def get_inline_comments(self) -> list[ReviewCommentSchema]:
        ...

    async def create_general_comment(self, message: str) -> None:
        ...

    async def create_inline_comment(self, file: str, line: int, message: str) -> None:
        ...

    async def update_general_comment(self, comment_id: int | str, message: str) -> None:
        ...

    async def delete_general_comment(self, comment_id: int | str) -> None:
        ...

    async def delete_inline_comment(self, comment_id: int | str) -> None:
        ...

    async def create_inline_reply(self, thread_id: int | str, message: str) -> None:
        ...

    async def create_summary_reply(self, thread_id: int | str, message: str) -> None:
        ...

    async def get_inline_threads(self) -> list[ReviewThreadSchema]:
        ...

    async def get_general_threads(self) -> list[ReviewThreadSchema]:
        ...
