import asyncio

import typer

from ai_review.cli.commands.run_clear_inline_review import run_clear_inline_review
from ai_review.cli.commands.run_clear_summary_review import run_clear_summary_review
from ai_review.cli.commands.run_context_review import run_context_review_command
from ai_review.cli.commands.run_inline_reply_review import run_inline_reply_review_command
from ai_review.cli.commands.run_inline_review import run_inline_review_command
from ai_review.cli.commands.run_review import run_review_command
from ai_review.cli.commands.run_summary_reply_review import run_summary_reply_review_command
from ai_review.cli.commands.run_summary_review import run_summary_review_command
from ai_review.config import settings
from ai_review.services.review.runner.outcome import ReviewOutcome

app = typer.Typer(help="AI Review CLI")


@app.command("run", help="Run the full AI review pipeline")
def run():
    typer.secho("Starting full AI review...", fg=typer.colors.CYAN, bold=True)
    inline_outcome, summary_outcome = asyncio.run(run_review_command())
    if settings.review.fail_on_empty_result and ReviewOutcome.EMPTY in (inline_outcome, summary_outcome):
        typer.secho("AI review produced no usable result", fg=typer.colors.RED, bold=True)
        raise typer.Exit(code=1)
    typer.secho("AI review completed successfully!", fg=typer.colors.GREEN, bold=True)


@app.command("run-inline", help="Run only the inline review")
def run_inline():
    typer.secho("Starting inline AI review...", fg=typer.colors.CYAN)
    outcome = asyncio.run(run_inline_review_command())
    if outcome == ReviewOutcome.EMPTY and settings.review.fail_on_empty_result:
        typer.secho("Inline AI review produced no usable result", fg=typer.colors.RED, bold=True)
        raise typer.Exit(code=1)
    typer.secho("AI review completed successfully!", fg=typer.colors.GREEN, bold=True)


@app.command("run-context", help="Run only the context review")
def run_context():
    typer.secho("Starting context AI review...", fg=typer.colors.CYAN)
    asyncio.run(run_context_review_command())
    typer.secho("AI review completed successfully!", fg=typer.colors.GREEN, bold=True)


@app.command("run-summary", help="Run only the summary review")
def run_summary():
    typer.secho("Starting summary AI review...", fg=typer.colors.CYAN)
    outcome = asyncio.run(run_summary_review_command())
    if outcome == ReviewOutcome.EMPTY and settings.review.fail_on_empty_result:
        typer.secho("Summary AI review produced no usable result", fg=typer.colors.RED, bold=True)
        raise typer.Exit(code=1)
    typer.secho("AI review completed successfully!", fg=typer.colors.GREEN, bold=True)


@app.command("run-inline-reply", help="Run only the inline reply review")
def run_inline_reply():
    typer.secho("Starting inline reply AI review...", fg=typer.colors.CYAN)
    asyncio.run(run_inline_reply_review_command())
    typer.secho("AI review completed successfully!", fg=typer.colors.GREEN, bold=True)


@app.command("run-summary-reply")
def run_summary_reply():
    typer.secho("Starting summary reply AI review...", fg=typer.colors.CYAN)
    asyncio.run(run_summary_reply_review_command())
    typer.secho("AI review completed successfully!", fg=typer.colors.GREEN, bold=True)


@app.command("clear-inline", help="Remove all AI-generated inline review comments")
def clear_inline():
    typer.secho("Clearing inline AI review comments...", fg=typer.colors.YELLOW)
    asyncio.run(run_clear_inline_review())
    typer.secho("Inline AI comments cleared", fg=typer.colors.GREEN, bold=True)


@app.command("clear-summary", help="Remove all AI-generated summary review comments")
def clear_summary():
    typer.secho("Clearing summary AI review comments...", fg=typer.colors.YELLOW)
    asyncio.run(run_clear_summary_review())
    typer.secho("Summary AI comments cleared", fg=typer.colors.GREEN, bold=True)


@app.command("show-config", help="Show the current resolved configuration")
def show_config():
    typer.secho("Loaded AI Review configuration:", fg=typer.colors.CYAN, bold=True)
    typer.echo(settings.model_dump_json(indent=2, exclude_none=True))


if __name__ == "__main__":
    app()
