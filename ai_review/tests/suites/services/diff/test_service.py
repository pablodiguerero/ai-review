import pytest

from ai_review import config
from ai_review.libs.config.review import ReviewMode
from ai_review.libs.diff.models import Diff, DiffFile, FileMode
from ai_review.services.diff.schema import DiffFileSchema
from ai_review.services.diff.service import DiffService
from ai_review.services.git.types import GitServiceProtocol
from ai_review.tests.fixtures.services.git import FakeGitService


class PerFileGitService(GitServiceProtocol):
    def __init__(self, diffs: dict[str, str]):
        self.diffs = diffs

    def get_diff(self, base_sha: str, head_sha: str, unified: int = 3) -> str:
        return ""

    def get_diff_for_file(self, base_sha: str, head_sha: str, file: str, unified: int = 3) -> str:
        return self.diffs.get(file, "")

    def get_changed_files(self, base_sha: str, head_sha: str) -> list[str]:
        return list(self.diffs)

    def get_file_at_commit(self, file_path: str, sha: str) -> str | None:
        return None


@pytest.fixture
def passthrough_render_file(monkeypatch: pytest.MonkeyPatch) -> None:
    def _render_file(
            cls: type[DiffService],
            file: str,
            raw_diff: str,
            base_sha: str | None = None,
            head_sha: str | None = None,
    ) -> DiffFileSchema:
        return DiffFileSchema(file=file, diff=raw_diff)

    monkeypatch.setattr(DiffService, "render_file", classmethod(_render_file))


@pytest.fixture
def fake_diff_file() -> DiffFile:
    return DiffFile(
        header="diff --git a/x b/x",
        mode=FileMode.MODIFIED,
        orig_name="a/x",
        new_name="b/x",
        hunks=[]
    )


@pytest.fixture
def fake_diff(fake_diff_file: DiffFile) -> Diff:
    return Diff(files=[fake_diff_file], raw="raw-diff")


def test_parse_empty_returns_empty_diff():
    diff = DiffService.parse("")
    assert diff.files == []
    assert diff.raw == ""


def test_parse_nonempty(monkeypatch: pytest.MonkeyPatch, fake_diff: Diff):
    monkeypatch.setattr("ai_review.services.diff.service.DiffParser.parse", lambda _: fake_diff)
    diff = DiffService.parse("something")
    assert diff.files[0].new_name == "b/x"


@pytest.mark.parametrize("mode,expected_prefix", [
    (ReviewMode.FULL_FILE_CURRENT, "# Failed to read current snapshot"),
    (ReviewMode.FULL_FILE_PREVIOUS, "# Failed to read previous snapshot"),
    (ReviewMode.FULL_FILE_DIFF, "# No matching lines for mode"),
    (ReviewMode.ONLY_ADDED, "# No matching lines for mode"),
    (ReviewMode.ONLY_REMOVED, "# No matching lines for mode"),
    (ReviewMode.ADDED_AND_REMOVED, "# No matching lines for mode"),
    (ReviewMode.ONLY_ADDED_WITH_CONTEXT, "# No matching lines for mode"),
    (ReviewMode.ONLY_REMOVED_WITH_CONTEXT, "# No matching lines for mode"),
    (ReviewMode.ADDED_AND_REMOVED_WITH_CONTEXT, "# No matching lines for mode"),
])
def test_render_file_routes_to_right_builder(
        mode: ReviewMode,
        fake_diff: Diff,
        monkeypatch: pytest.MonkeyPatch,
        expected_prefix: str
):
    monkeypatch.setattr("ai_review.services.diff.service.DiffParser.parse", lambda _: fake_diff)
    monkeypatch.setattr(config.settings.review, "mode", mode)

    out = DiffService.render_file(raw_diff="fake", file="b/x")
    assert out.file == "b/x"
    assert out.diff.startswith(expected_prefix)


def test_render_file_returns_unsupported(monkeypatch: pytest.MonkeyPatch, fake_diff: Diff):
    monkeypatch.setattr("ai_review.services.diff.service.DiffParser.parse", lambda _: fake_diff)
    monkeypatch.setattr(config.settings.review, "mode", "NON_EXISTING")
    out = DiffService.render_file(raw_diff="fake", file="b/x")
    assert out.file == "b/x"
    assert "# Unsupported mode" in out.diff


def test_render_files_invokes_render_file(
        fake_diff: Diff,
        monkeypatch: pytest.MonkeyPatch,
        fake_git_service: FakeGitService,
) -> None:
    monkeypatch.setattr("ai_review.services.diff.service.DiffParser.parse", lambda _: fake_diff)
    monkeypatch.setattr(config.settings.review, "mode", ReviewMode.FULL_FILE_DIFF)

    fake_git_service.responses["get_diff_for_file"] = "fake-diff"

    out = DiffService.render_files(git=fake_git_service, base_sha="A", head_sha="B", files=["b/x"])
    assert out
    assert out[0].file == "b/x"
    assert out[0].diff.startswith("# No matching lines for mode")



def test_render_files_under_budget_is_unchanged(
        monkeypatch: pytest.MonkeyPatch,
        passthrough_render_file: None,
) -> None:
    monkeypatch.setattr(config.settings.review, "max_diff_chars", 1000)
    git = PerFileGitService({"a.py": "a" * 100, "b.py": "b" * 100})

    out = DiffService.render_files(git=git, base_sha="A", head_sha="B", files=["a.py", "b.py"])

    assert [entry.file for entry in out] == ["a.py", "b.py"]
    assert out[0].diff == "a" * 100
    assert out[1].diff == "b" * 100


def test_render_files_over_budget_keeps_fitting_files_and_appends_notice(
        monkeypatch: pytest.MonkeyPatch,
        passthrough_render_file: None,
) -> None:
    monkeypatch.setattr(config.settings.review, "max_diff_chars", 150)
    git = PerFileGitService({"a.py": "a" * 100, "b.py": "b" * 100, "c.py": "c" * 100})

    out = DiffService.render_files(git=git, base_sha="A", head_sha="B", files=["a.py", "b.py", "c.py"])

    assert [entry.file for entry in out] == ["a.py", "ai-review: truncation notice"]
    assert out[0].diff == "a" * 100

    notice = out[-1].diff
    assert "Only 1 of 3 changed files are shown below" in notice
    assert "2 files omitted" in notice
    assert "REVIEW__MAX_DIFF_CHARS=150" in notice


def test_render_files_oversized_single_file_is_truncated_with_notice(
        monkeypatch: pytest.MonkeyPatch,
        passthrough_render_file: None,
) -> None:
    monkeypatch.setattr(config.settings.review, "max_diff_chars", 100)
    git = PerFileGitService({"a.py": "a" * 500})

    out = DiffService.render_files(git=git, base_sha="A", head_sha="B", files=["a.py"])

    assert [entry.file for entry in out] == ["a.py", "ai-review: truncation notice"]
    assert len(out[0].diff) == 100
    assert out[0].diff.endswith("... file diff truncated ...")

    notice = out[-1].diff
    assert "Only 1 of 1 changed files are shown below" in notice
    assert "0 files omitted" in notice
    assert "REVIEW__MAX_DIFF_CHARS=100" in notice


def test_render_files_max_diff_chars_none_disables_cap(
        monkeypatch: pytest.MonkeyPatch,
        passthrough_render_file: None,
) -> None:
    monkeypatch.setattr(config.settings.review, "max_diff_chars", None)
    git = PerFileGitService({"a.py": "a" * 10_000, "b.py": "b" * 10_000})

    out = DiffService.render_files(git=git, base_sha="A", head_sha="B", files=["a.py", "b.py"])

    assert [entry.file for entry in out] == ["a.py", "b.py"]
    assert len(out[0].diff) == 10_000
    assert len(out[1].diff) == 10_000


def test_render_batches_single_batch_matches_render_files_when_under_budget(
        monkeypatch: pytest.MonkeyPatch,
        passthrough_render_file: None,
) -> None:
    monkeypatch.setattr(config.settings.review, "max_diff_chars", 1000)
    git = PerFileGitService({"a.py": "a" * 100, "b.py": "b" * 100})

    out = DiffService.render_batches(git=git, base_sha="A", head_sha="B", files=["a.py", "b.py"])

    assert len(out) == 1
    assert [entry.file for entry in out[0]] == ["a.py", "b.py"]
    assert out[0][0].diff == "a" * 100
    assert out[0][1].diff == "b" * 100


def test_render_batches_splits_whole_files_across_batches(
        monkeypatch: pytest.MonkeyPatch,
        passthrough_render_file: None,
) -> None:
    monkeypatch.setattr(config.settings.review, "max_diff_chars", 150)
    git = PerFileGitService({"a.py": "a" * 100, "b.py": "b" * 100, "c.py": "c" * 100})

    out = DiffService.render_batches(git=git, base_sha="A", head_sha="B", files=["a.py", "b.py", "c.py"])

    assert [[entry.file for entry in batch] for batch in out] == [["a.py"], ["b.py"], ["c.py"]]
    assert out[0][0].diff == "a" * 100
    assert out[1][0].diff == "b" * 100
    assert out[2][0].diff == "c" * 100


def test_render_batches_oversized_single_file_becomes_its_own_truncated_batch(
        monkeypatch: pytest.MonkeyPatch,
        passthrough_render_file: None,
) -> None:
    monkeypatch.setattr(config.settings.review, "max_diff_chars", 100)
    git = PerFileGitService({"a.py": "a" * 500, "b.py": "b" * 50})

    out = DiffService.render_batches(git=git, base_sha="A", head_sha="B", files=["a.py", "b.py"])

    assert [[entry.file for entry in batch] for batch in out] == [["a.py"], ["b.py"]]
    assert len(out[0][0].diff) == 100
    assert out[0][0].diff.endswith("... file diff truncated ...")
    assert out[1][0].diff == "b" * 50


def test_render_batches_caps_batch_count_and_appends_coverage_notice(
        monkeypatch: pytest.MonkeyPatch,
        passthrough_render_file: None,
) -> None:
    monkeypatch.setattr(config.settings.review, "max_diff_chars", 100)
    monkeypatch.setattr(config.settings.review, "max_diff_batches", 2)
    git = PerFileGitService({name: "x" * 100 for name in ["a.py", "b.py", "c.py", "d.py", "e.py"]})

    out = DiffService.render_batches(
        git=git, base_sha="A", head_sha="B", files=["a.py", "b.py", "c.py", "d.py", "e.py"]
    )

    assert len(out) == 2
    assert [entry.file for entry in out[0]] == ["a.py"]
    assert [entry.file for entry in out[1]] == ["b.py", "ai-review: coverage notice"]

    notice = out[1][-1].diff
    assert "3 of 5 files not reviewed" in notice
    assert "REVIEW__MAX_DIFF_BATCHES=2" in notice
    assert "REVIEW__MAX_DIFF_CHARS=100" in notice


def test_render_batches_max_diff_chars_none_returns_single_batch(
        monkeypatch: pytest.MonkeyPatch,
        passthrough_render_file: None,
) -> None:
    monkeypatch.setattr(config.settings.review, "max_diff_chars", None)
    git = PerFileGitService({"a.py": "a" * 10_000, "b.py": "b" * 10_000})

    out = DiffService.render_batches(git=git, base_sha="A", head_sha="B", files=["a.py", "b.py"])

    assert len(out) == 1
    assert [entry.file for entry in out[0]] == ["a.py", "b.py"]


def test_render_batches_no_files_returns_empty_list(
        monkeypatch: pytest.MonkeyPatch,
        passthrough_render_file: None,
) -> None:
    monkeypatch.setattr(config.settings.review, "max_diff_chars", 100)
    git = PerFileGitService({})

    out = DiffService.render_batches(git=git, base_sha="A", head_sha="B", files=[])

    assert out == []
