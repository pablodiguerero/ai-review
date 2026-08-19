from ai_review.config import settings
from ai_review.services.agent.loop.schema import AgentTraceSchema
from ai_review.services.diff.schema import DiffFileSchema
from ai_review.services.prompt.schema import PromptContextSchema
from ai_review.services.prompt.tools import (
    format_file,
    format_files,
    format_thread,
    format_traces,
    normalize_prompt,
)
from ai_review.services.prompt.types import PromptServiceProtocol
from ai_review.services.vcs.types import ReviewThreadSchema


class PromptService(PromptServiceProtocol):
    @classmethod
    def prepare_prompt(cls, prompts: list[str], context: PromptContextSchema) -> str:
        prompt = "\n\n".join(prompts)
        prompt = context.apply_format(prompt)

        if settings.prompt.normalize_prompts:
            prompt = normalize_prompt(prompt)

        return prompt

    @classmethod
    def build_agent_request(
            cls,
            traces: list[AgentTraceSchema],
            force_final: bool,
            original_prompt: str,
            original_prompt_system: str,
    ) -> str:
        force_final_mode = (
            "Return FINAL only. The tool budget is exhausted: do NOT request any TOOL_CALL and do NOT "
            "deliberate about further verification. Write the final task output now from the evidence "
            "already present in the agent history; if a claim could not be verified, state that "
            "explicitly and mark it low-confidence instead of withholding the review. Keep any reasoning "
            "brief and put the complete review into the FINAL `content` string."
        )
        mode = force_final_mode if force_final else "You can either call a tool or return FINAL."
        history = format_traces(traces)
        agent_prompt = cls.prepare_prompt(settings.prompt.load_agent(), PromptContextSchema())

        return (
            f"{agent_prompt}\n\n"
            f"## Agent mode\n{mode}\n\n"
            f"## Task output format\n{original_prompt_system}\n\n"
            f"## Task\n{original_prompt}\n\n"
            f"## Agent history\n{history}\n\n"
        )

    @classmethod
    def build_inline_request(cls, diff: DiffFileSchema, context: PromptContextSchema) -> str:
        prompt = cls.prepare_prompt(settings.prompt.load_inline(), context)
        return (
            f"{prompt}\n\n"
            f"## Diff\n\n"
            f"{format_file(diff)}"
        )

    @classmethod
    def build_summary_request(
            cls,
            diffs: list[DiffFileSchema],
            context: PromptContextSchema,
            prior_feedback: str | None = None,
    ) -> str:
        prompt = cls.prepare_prompt(settings.prompt.load_summary(), context)
        changes = format_files(diffs)
        feedback = (
            f"## Previous reviews and team responses\n\n{prior_feedback}\n\n"
            if prior_feedback
            else ""
        )
        return (
            f"{prompt}\n\n"
            f"{feedback}"
            f"## Changes\n\n"
            f"{changes}\n"
        )

    @classmethod
    def build_context_request(cls, diffs: list[DiffFileSchema], context: PromptContextSchema) -> str:
        prompt = cls.prepare_prompt(settings.prompt.load_context(), context)
        changes = format_files(diffs)
        return (
            f"{prompt}\n\n"
            f"## Diff\n\n"
            f"{changes}\n"
        )

    @classmethod
    def build_inline_reply_request(
            cls,
            diff: DiffFileSchema,
            thread: ReviewThreadSchema,
            context: PromptContextSchema
    ) -> str:
        prompt = cls.prepare_prompt(settings.prompt.load_inline_reply(), context)
        conversation = format_thread(thread)

        return (
            f"{prompt}\n\n"
            f"## Conversation\n\n"
            f"{conversation}\n\n"
            f"## Diff\n\n"
            f"{format_file(diff)}"
        )

    @classmethod
    def build_summary_reply_request(
            cls,
            diffs: list[DiffFileSchema],
            thread: ReviewThreadSchema,
            context: PromptContextSchema
    ) -> str:
        prompt = cls.prepare_prompt(settings.prompt.load_summary_reply(), context)
        changes = format_files(diffs)
        conversation = format_thread(thread)

        return (
            f"{prompt}\n\n"
            f"## Conversation\n\n"
            f"{conversation}\n\n"
            f"## Changes\n\n"
            f"{changes}"
        )

    @classmethod
    def build_system_agent_request(cls) -> str:
        return cls.prepare_prompt(settings.prompt.load_system_agent(), PromptContextSchema())

    @classmethod
    def build_system_inline_request(cls, context: PromptContextSchema) -> str:
        return cls.prepare_prompt(settings.prompt.load_system_inline(), context)

    @classmethod
    def build_system_context_request(cls, context: PromptContextSchema) -> str:
        return cls.prepare_prompt(settings.prompt.load_system_context(), context)

    @classmethod
    def build_system_summary_request(cls, context: PromptContextSchema) -> str:
        return cls.prepare_prompt(settings.prompt.load_system_summary(), context)

    @classmethod
    def build_system_inline_reply_request(cls, context: PromptContextSchema) -> str:
        return cls.prepare_prompt(settings.prompt.load_system_inline_reply(), context)

    @classmethod
    def build_system_summary_reply_request(cls, context: PromptContextSchema) -> str:
        return cls.prepare_prompt(settings.prompt.load_system_summary_reply(), context)
