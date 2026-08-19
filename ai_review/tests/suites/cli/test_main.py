import pytest
from typer.testing import CliRunner

from ai_review.cli.main import app
from ai_review.config import settings
from ai_review.services.review.runner.outcome import ReviewOutcome
from ai_review.services.review.service import ReviewService
from ai_review.tests.fixtures.services.review.runner.inline import FakeInlineReviewRunner
from ai_review.tests.fixtures.services.review.runner.summary import FakeSummaryReviewRunner

runner = CliRunner()


@pytest.fixture(autouse=True)
def dummy_review_service(monkeypatch: pytest.MonkeyPatch, review_service: ReviewService):
    monkeypatch.setattr("ai_review.cli.commands.run_review.ReviewService", lambda: review_service)
    monkeypatch.setattr("ai_review.cli.commands.run_inline_review.ReviewService", lambda: review_service)
    monkeypatch.setattr("ai_review.cli.commands.run_context_review.ReviewService", lambda: review_service)
    monkeypatch.setattr("ai_review.cli.commands.run_summary_review.ReviewService", lambda: review_service)
    monkeypatch.setattr("ai_review.cli.commands.run_inline_reply_review.ReviewService", lambda: review_service)
    monkeypatch.setattr("ai_review.cli.commands.run_summary_reply_review.ReviewService", lambda: review_service)


@pytest.mark.parametrize(
    "args, expected_output",
    [
        (["run"], "Starting full AI review..."),
        (["run-inline"], "Starting inline AI review..."),
        (["run-context"], "Starting context AI review..."),
        (["run-summary"], "Starting summary AI review..."),
        (["run-inline-reply"], "Starting inline reply AI review..."),
        (["run-summary-reply"], "Starting summary reply AI review..."),
    ],
)
def test_cli_commands_invoke_review_service_successfully(args: list[str], expected_output: str):
    result = runner.invoke(app, args)

    assert result.exit_code == 0
    assert expected_output in result.output
    assert "AI review completed successfully!" in result.output


def test_show_config_outputs_json(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "ai_review.cli.main.settings.model_dump_json",
        lambda **_: '{"debug": true}'
    )

    result = runner.invoke(app, ["show-config"])
    assert result.exit_code == 0
    assert "Loaded AI Review configuration" in result.output
    assert '{"debug": true}' in result.output


def test_run_inline_exits_with_error_when_empty_and_fail_on_empty_result_enabled(
        monkeypatch: pytest.MonkeyPatch,
        fake_inline_review_runner: FakeInlineReviewRunner,
):
    monkeypatch.setattr(settings.review, "fail_on_empty_result", True)
    fake_inline_review_runner.outcome = ReviewOutcome.EMPTY

    result = runner.invoke(app, ["run-inline"])

    assert result.exit_code == 1
    assert "produced no usable result" in result.output
    assert "AI review completed successfully!" not in result.output


def test_run_inline_exits_zero_when_empty_and_fail_on_empty_result_disabled(
        monkeypatch: pytest.MonkeyPatch,
        fake_inline_review_runner: FakeInlineReviewRunner,
):
    monkeypatch.setattr(settings.review, "fail_on_empty_result", False)
    fake_inline_review_runner.outcome = ReviewOutcome.EMPTY

    result = runner.invoke(app, ["run-inline"])

    assert result.exit_code == 0
    assert "AI review completed successfully!" in result.output


def test_run_summary_exits_with_error_when_empty_and_fail_on_empty_result_enabled(
        monkeypatch: pytest.MonkeyPatch,
        fake_summary_review_runner: FakeSummaryReviewRunner,
):
    monkeypatch.setattr(settings.review, "fail_on_empty_result", True)
    fake_summary_review_runner.outcome = ReviewOutcome.EMPTY

    result = runner.invoke(app, ["run-summary"])

    assert result.exit_code == 1
    assert "produced no usable result" in result.output
    assert "AI review completed successfully!" not in result.output


def test_run_summary_exits_zero_when_posted_even_with_fail_on_empty_result_enabled(
        monkeypatch: pytest.MonkeyPatch,
        fake_summary_review_runner: FakeSummaryReviewRunner,
):
    monkeypatch.setattr(settings.review, "fail_on_empty_result", True)
    fake_summary_review_runner.outcome = ReviewOutcome.POSTED

    result = runner.invoke(app, ["run-summary"])

    assert result.exit_code == 0
    assert "AI review completed successfully!" in result.output


def test_run_exits_with_error_when_inline_empty_and_fail_on_empty_result_enabled(
        monkeypatch: pytest.MonkeyPatch,
        fake_inline_review_runner: FakeInlineReviewRunner,
        fake_summary_review_runner: FakeSummaryReviewRunner,
):
    monkeypatch.setattr(settings.review, "fail_on_empty_result", True)
    fake_inline_review_runner.outcome = ReviewOutcome.EMPTY
    fake_summary_review_runner.outcome = ReviewOutcome.POSTED

    result = runner.invoke(app, ["run"])

    assert result.exit_code == 1
    assert "produced no usable result" in result.output
    assert "AI review completed successfully!" not in result.output


def test_run_exits_with_error_when_summary_empty_and_fail_on_empty_result_enabled(
        monkeypatch: pytest.MonkeyPatch,
        fake_inline_review_runner: FakeInlineReviewRunner,
        fake_summary_review_runner: FakeSummaryReviewRunner,
):
    monkeypatch.setattr(settings.review, "fail_on_empty_result", True)
    fake_inline_review_runner.outcome = ReviewOutcome.POSTED
    fake_summary_review_runner.outcome = ReviewOutcome.EMPTY

    result = runner.invoke(app, ["run"])

    assert result.exit_code == 1
    assert "produced no usable result" in result.output
    assert "AI review completed successfully!" not in result.output


def test_run_exits_zero_when_both_posted_and_fail_on_empty_result_enabled(
        monkeypatch: pytest.MonkeyPatch,
        fake_inline_review_runner: FakeInlineReviewRunner,
        fake_summary_review_runner: FakeSummaryReviewRunner,
):
    monkeypatch.setattr(settings.review, "fail_on_empty_result", True)
    fake_inline_review_runner.outcome = ReviewOutcome.POSTED
    fake_summary_review_runner.outcome = ReviewOutcome.POSTED

    result = runner.invoke(app, ["run"])

    assert result.exit_code == 0
    assert "AI review completed successfully!" in result.output


def test_run_exits_zero_when_empty_and_fail_on_empty_result_disabled(
        monkeypatch: pytest.MonkeyPatch,
        fake_inline_review_runner: FakeInlineReviewRunner,
        fake_summary_review_runner: FakeSummaryReviewRunner,
):
    monkeypatch.setattr(settings.review, "fail_on_empty_result", False)
    fake_inline_review_runner.outcome = ReviewOutcome.EMPTY
    fake_summary_review_runner.outcome = ReviewOutcome.EMPTY

    result = runner.invoke(app, ["run"])

    assert result.exit_code == 0
    assert "AI review completed successfully!" in result.output
