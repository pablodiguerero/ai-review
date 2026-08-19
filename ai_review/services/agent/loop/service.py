import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from ai_review.config import settings
from ai_review.libs.llm.output_json_parser import LLMOutputJSONParser
from ai_review.libs.logger import get_logger
from ai_review.services.agent.checkpoint.schema import AgentCheckpointSchema
from ai_review.services.agent.checkpoint.service import AgentCheckpointService
from ai_review.services.agent.checkpoint.types import AgentCheckpointServiceProtocol
from ai_review.services.agent.loop.schema import (
    AgentAction,
    AgentStepSchema,
    AgentTraceSchema,
    AgentLoopResultSchema
)
from ai_review.services.agent.loop.types import AgentLoopServiceProtocol
from ai_review.services.agent.tool.types import AgentToolServiceProtocol
from ai_review.services.llm.types import LLMClientProtocol, ChatResultSchema
from ai_review.services.prompt.types import PromptServiceProtocol

logger = get_logger("AGENT_LOOP_SERVICE")


class AgentVerificationAborted(Exception):
    pass


@dataclass
class AgentRunState:
    traces: list[AgentTraceSchema] = field(default_factory=list)
    signatures: set[str] = field(default_factory=set)
    context_used: int = 0
    iterations: int = 0
    executed_tool_calls: int = 0
    blocked_tool_calls: int = 0
    start_time: float = 0.0
    deadline_at: float | None = None


class AgentLoopService(AgentLoopServiceProtocol):
    def __init__(
            self,
            llm: LLMClientProtocol,
            prompt: PromptServiceProtocol,
            agent_tool: AgentToolServiceProtocol,
            checkpoint: AgentCheckpointServiceProtocol | None = None,
            clock: Callable[[], float] = time.monotonic,
    ):
        self.llm = llm
        self.prompt = prompt
        self.agent_tool = agent_tool
        self.checkpoint = checkpoint or AgentCheckpointService()
        self.clock = clock
        self.max_iterations = settings.agent.max_iterations
        self.max_context_chars = settings.agent.max_total_context_chars
        self.min_tool_calls = settings.agent.min_tool_calls
        self.empty_response_retries = settings.agent.empty_response_retries
        self.force_final_attempts = settings.agent.force_final_attempts
        self.deadline_seconds = settings.agent.deadline_seconds
        self.llm_request_timeout = settings.llm.http_client.timeout

        self.parser = LLMOutputJSONParser(AgentStepSchema)

    def _log_summary(self, state: AgentRunState, stop_reason: str) -> None:
        logger.info(
            f"Agent loop summary: iterations={state.iterations} "
            f"executed_tool_calls={state.executed_tool_calls} "
            f"blocked_tool_calls={state.blocked_tool_calls} stop_reason={stop_reason}"
        )

    async def _save_checkpoint(
            self,
            key: str,
            state: AgentRunState,
            finished_iterations: bool,
            traces: list[AgentTraceSchema] | None = None,
    ) -> None:
        checkpoint = AgentCheckpointSchema(
            key=key,
            traces=traces if traces is not None else list(state.traces),
            executed_tool_calls=state.executed_tool_calls,
            blocked_tool_calls=state.blocked_tool_calls,
            iterations=state.iterations,
            context_used=state.context_used,
            signatures=list(state.signatures),
            finished_iterations=finished_iterations,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        await self.checkpoint.save(key, checkpoint)

    async def _chat(
            self,
            prompt: str,
            prompt_system: str,
            deadline_at: float | None = None,
    ) -> ChatResultSchema:
        result = await self.llm.chat(prompt=prompt, prompt_system=prompt_system, json_mode=True)
        for attempt in range(1, self.empty_response_retries + 1):
            if (result.text or "").strip():
                return result

            if deadline_at is not None and self.clock() >= deadline_at:
                logger.warning("Agent loop deadline reached; skipping empty-response retry")
                return result

            logger.warning(f"LLM returned empty content; retrying ({attempt}/{self.empty_response_retries})")
            result = await self.llm.chat(prompt=prompt, prompt_system=prompt_system, json_mode=True)
        return result

    async def run_step(
            self,
            state: AgentRunState,
            step: AgentStepSchema,
            chat: ChatResultSchema,
            iteration: int,
    ) -> tuple[AgentTraceSchema, bool]:
        if step.command in state.signatures:
            logger.debug(f"Duplicate tool call blocked at iteration {iteration}: {step.command}")
            trace = AgentTraceSchema(
                step=step,
                warning=f"Duplicate tool call blocked: {step.command}",
                iteration=iteration,
                raw_output=chat.text,
                total_tokens=chat.total_tokens,
                prompt_tokens=chat.prompt_tokens,
                completion_tokens=chat.completion_tokens,
            )
            return trace, False

        state.signatures.add(step.command)
        logger.debug(f"Executing agent tool command at iteration {iteration}: {step.command}")
        tool_result = await self.agent_tool.execute(step.command)

        trace = AgentTraceSchema(
            step=step,
            iteration=iteration,
            raw_output=chat.text,
            tool_output=tool_result.output,
            total_tokens=chat.total_tokens,
            prompt_tokens=chat.prompt_tokens,
            completion_tokens=chat.completion_tokens,
        )
        return trace, tool_result.executed

    async def force_final(
            self,
            state: AgentRunState,
            prompt: str,
            prompt_system: str,
    ) -> AgentLoopResultSchema:
        if state.executed_tool_calls < self.min_tool_calls:
            logger.error(
                f"Force-final skipped: only {state.executed_tool_calls}/{self.min_tool_calls} "
                "verification commands run; refusing to publish an unverified review"
            )
            return AgentLoopResultSchema(
                traces=state.traces,
                final_text="",
                stop_reason="forced_final_no_review",
                iterations=state.iterations,
                executed_tool_calls=state.executed_tool_calls,
                blocked_tool_calls=state.blocked_tool_calls,
            )

        logger.info("Forcing FINAL response after loop limits reached")

        agent_prompt_system = self.prompt.build_system_agent_request()
        last_result: ChatResultSchema | None = None
        last_error: Exception | None = None

        for attempt in range(1, self.force_final_attempts + 1):
            if (
                    self.deadline_seconds is not None
                    and attempt > 1
                    and (self.clock() - state.start_time) >= (
                            self.deadline_seconds + self.llm_request_timeout * (attempt - 1)
                    )
            ):
                logger.info(
                    f"Force-final attempt {attempt}/{self.force_final_attempts} skipped: "
                    "past the deadline request budget"
                )
                break

            agent_prompt = self.prompt.build_agent_request(
                traces=state.traces,
                force_final=True,
                original_prompt=prompt,
                original_prompt_system=prompt_system,
            )

            try:
                last_result = await self._chat(agent_prompt, agent_prompt_system, deadline_at=state.deadline_at)
            except Exception as error:
                last_error = error
                logger.warning(f"Force-final attempt {attempt}/{self.force_final_attempts} failed: {error}")
                continue

            last_error = None
            step = self.parser.parse_output(last_result.text)
            is_final = bool(step and step.action.is_final and (step.content or "").strip())
            logger.debug(
                f"Force-final attempt {attempt}/{self.force_final_attempts}; parsed_as_final={is_final}"
            )

            if is_final:
                state.traces.append(
                    AgentTraceSchema(
                        step=step,
                        warning="Forced final response after loop limits reached.",
                        iteration=len(state.traces) + 1,
                        raw_output=last_result.text,
                        total_tokens=last_result.total_tokens,
                        prompt_tokens=last_result.prompt_tokens,
                        completion_tokens=last_result.completion_tokens,
                    )
                )
                return AgentLoopResultSchema(
                    traces=state.traces,
                    final_text=step.content,
                    stop_reason="forced_final",
                    iterations=state.iterations,
                    executed_tool_calls=state.executed_tool_calls,
                    blocked_tool_calls=state.blocked_tool_calls,
                )

            if step is not None:
                state.traces.append(
                    AgentTraceSchema(
                        step=step,
                        warning=(
                            "REJECTED: force-final requires a FINAL action carrying your review in "
                            "`content`. The loop is over — do NOT call tools now, return FINAL."
                        ),
                        iteration=len(state.traces) + 1,
                        raw_output=last_result.text,
                        total_tokens=last_result.total_tokens,
                        prompt_tokens=last_result.prompt_tokens,
                        completion_tokens=last_result.completion_tokens,
                    )
                )

        if last_result is None and last_error is not None:
            raise last_error

        logger.warning(
            f"Force-final produced no valid FINAL after {self.force_final_attempts} attempt(s); "
            f"skipping summary (no comment will be posted)"
        )
        state.traces.append(
            AgentTraceSchema(
                step=AgentStepSchema(
                    action=AgentAction.FINAL,
                    content="Agent could not produce a valid FINAL review after force-final.",
                ),
                warning="Force-final exhausted without a valid FINAL; no summary posted.",
                iteration=len(state.traces) + 1,
                raw_output=(last_result.text if last_result else ""),
                total_tokens=(last_result.total_tokens if last_result else None),
                prompt_tokens=(last_result.prompt_tokens if last_result else None),
                completion_tokens=(last_result.completion_tokens if last_result else None),
            )
        )
        return AgentLoopResultSchema(
            traces=state.traces,
            final_text="",
            stop_reason="forced_final_no_review",
            iterations=state.iterations,
            executed_tool_calls=state.executed_tool_calls,
            blocked_tool_calls=state.blocked_tool_calls,
        )

    async def run(
            self,
            prompt: str,
            prompt_system: str,
            checkpoint_key: str | None = None,
    ) -> AgentLoopResultSchema:
        state = AgentRunState()
        resume_from_iteration = 1
        finished_iterations = False

        if checkpoint_key:
            restored = await self.checkpoint.load(checkpoint_key)
            if restored is not None:
                state.traces = list(restored.traces)
                state.signatures = set(restored.signatures)
                state.executed_tool_calls = restored.executed_tool_calls
                state.blocked_tool_calls = restored.blocked_tool_calls
                state.iterations = restored.iterations
                state.context_used = restored.context_used
                finished_iterations = restored.finished_iterations
                resume_from_iteration = restored.iterations + 1

                if finished_iterations:
                    logger.info(
                        "Agent loop resumed from checkpoint (finished_iterations=true): "
                        "going straight to force-final"
                    )
                else:
                    logger.info(
                        f"Agent loop resumed from checkpoint: iterations={restored.iterations} "
                        f"executed={restored.executed_tool_calls}"
                    )

        state.start_time = self.clock()
        if self.deadline_seconds is not None:
            state.deadline_at = state.start_time + self.deadline_seconds

        logger.info(
            f"Starting agent loop: max_iterations={self.max_iterations}, "
            f"min_tool_calls={self.min_tool_calls}, max_context_chars={self.max_context_chars}"
        )

        if finished_iterations:
            return await self._finish_with_force_final(state, checkpoint_key, prompt, prompt_system)

        for iteration in range(resume_from_iteration, self.max_iterations + 1):
            if state.deadline_at is not None and self.clock() >= state.deadline_at:
                logger.info(
                    f"Agent loop deadline reached ({self.deadline_seconds}s) before iteration {iteration}; "
                    "switching to force-final flow"
                )
                break

            logger.debug(f"Agent loop iteration started: {iteration}")

            agent_prompt = self.prompt.build_agent_request(
                traces=state.traces,
                force_final=False,
                original_prompt=prompt,
                original_prompt_system=prompt_system,
            )
            agent_prompt_system = self.prompt.build_system_agent_request()
            logger.debug(
                f"Agent prompt for iteration {iteration} "
                f"(prompt_chars={len(agent_prompt)}, system_chars={len(agent_prompt_system)}, "
                f"traces={len(state.traces)})"
            )

            state.iterations = iteration
            try:
                result = await self._chat(agent_prompt, agent_prompt_system, deadline_at=state.deadline_at)
            except Exception as error:
                if state.executed_tool_calls < self.min_tool_calls:
                    logger.error(
                        f"Agent loop iteration {iteration} failed ({error}) after only "
                        f"{state.executed_tool_calls}/{self.min_tool_calls} verification commands; aborting"
                    )
                    self._log_summary(state, "aborted")
                    raise AgentVerificationAborted(
                        f"Agent aborted after {state.executed_tool_calls}/{self.min_tool_calls} "
                        f"verification commands: {error}"
                    ) from error

                logger.warning(
                    f"Agent loop iteration {iteration} failed ({error}); "
                    f"switching to force-final flow with {len(state.traces)} traces kept"
                )
                break

            logger.debug(f"Agent LLM response at iteration {iteration}: {result.text[:500]}")

            step: AgentStepSchema | None = self.parser.parse_output(result.text)
            if step is None:
                logger.info(
                    f"Agent loop iteration {iteration} returned an unparseable response; "
                    f"switching to force-final flow"
                )
                break

            if step.action.is_final:
                if state.executed_tool_calls < self.min_tool_calls:
                    logger.info(
                        f"Iteration {iteration}: FINAL rejected — only "
                        f"{state.executed_tool_calls}/{self.min_tool_calls} "
                        f"verification commands run; nudging the agent to keep verifying"
                    )
                    state.traces.append(
                        AgentTraceSchema(
                            step=step,
                            warning=(
                                f"REJECTED: you returned FINAL after only {state.executed_tool_calls} "
                                f"verification command(s). You MUST run at least {self.min_tool_calls} "
                                f"read-only commands (e.g. grep/cat node_modules, rg src) to verify the "
                                f"diff's claims before finalizing. Do NOT finalize yet — issue a TOOL_CALL now."
                            ),
                            iteration=iteration,
                            raw_output=result.text,
                            total_tokens=result.total_tokens,
                            prompt_tokens=result.prompt_tokens,
                            completion_tokens=result.completion_tokens,
                        )
                    )
                    if checkpoint_key:
                        await self._save_checkpoint(checkpoint_key, state, finished_iterations=False)
                    continue

                logger.info(f"Agent loop iteration {iteration} returned FINAL action")
                if checkpoint_key:
                    await self._save_checkpoint(checkpoint_key, state, finished_iterations=True)

                state.traces.append(
                    AgentTraceSchema(
                        step=step,
                        iteration=iteration,
                        raw_output=result.text,
                        total_tokens=result.total_tokens,
                        prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                    )
                )

                self._log_summary(state, "final")
                return AgentLoopResultSchema(
                    traces=state.traces,
                    final_text=step.content,
                    stop_reason="final",
                    iterations=state.iterations,
                    executed_tool_calls=state.executed_tool_calls,
                    blocked_tool_calls=state.blocked_tool_calls,
                )

            trace, executed = await self.run_step(state=state, step=step, chat=result, iteration=iteration)
            state.traces.append(trace)
            if executed:
                state.executed_tool_calls += 1
            else:
                state.blocked_tool_calls += 1

            state.context_used += len(trace.tool_output or "")
            logger.debug(
                f"Agent loop context usage after iteration {iteration}: "
                f"{state.context_used}/{self.max_context_chars}"
            )
            if state.context_used >= self.max_context_chars:
                logger.info("Agent context limit reached, forcing final response")
                break

            if checkpoint_key:
                await self._save_checkpoint(checkpoint_key, state, finished_iterations=False)

        logger.info("Agent loop finished regular iterations without FINAL action; switching to force-final flow")
        return await self._finish_with_force_final(state, checkpoint_key, prompt, prompt_system)

    async def _finish_with_force_final(
            self,
            state: AgentRunState,
            checkpoint_key: str | None,
            prompt: str,
            prompt_system: str,
    ) -> AgentLoopResultSchema:
        if checkpoint_key:
            await self._save_checkpoint(checkpoint_key, state, finished_iterations=True)

        clean_traces = list(state.traces)
        result = await self.force_final(state=state, prompt=prompt, prompt_system=prompt_system)
        self._log_summary(state, result.stop_reason)

        if checkpoint_key and result.stop_reason == "forced_final":
            await self._save_checkpoint(checkpoint_key, state, finished_iterations=True, traces=clean_traces)

        return result
