import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from ai_review.config import settings
from ai_review.libs.llm.output_json_parser import LLMOutputJSONParser
from ai_review.libs.logger import get_logger
from ai_review.services.agent.checkpoint.schema import AgentCheckpointSchema, AgentCheckpointStage
from ai_review.services.agent.checkpoint.service import AgentCheckpointService
from ai_review.services.agent.checkpoint.types import AgentCheckpointServiceProtocol
from ai_review.services.agent.loop.schema import (
    AgentAction,
    AgentStepSchema,
    AgentTraceSchema,
    AgentLoopResultSchema
)
from ai_review.services.agent.loop.trace_trim import combine_synopsis, trim_traces
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
    new_executed: int = 0
    round: int = 0
    prior_synopsis: str = ""
    created_at: str = ""
    start_time: float = 0.0
    deadline_at: float | None = None
    final_replays: int = 0


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
        self.max_final_replays = settings.agent.max_final_replays
        self.deadline_seconds = settings.agent.deadline_seconds
        self.llm_request_timeout = settings.llm.http_client.timeout
        self.resume_min_new_tool_calls = settings.agent.resume_min_new_tool_calls
        self.max_trace_history = settings.agent.max_trace_history

        self.parser = LLMOutputJSONParser(AgentStepSchema)

    def _log_summary(self, state: AgentRunState, stop_reason: str) -> None:
        logger.info(
            f"Agent loop summary: iterations={state.iterations} "
            f"executed_tool_calls={state.executed_tool_calls} "
            f"blocked_tool_calls={state.blocked_tool_calls} stop_reason={stop_reason}"
        )

    def _fold_overflow(self, state: AgentRunState) -> None:
        if len(state.traces) <= self.max_trace_history:
            return

        kept, dropped_synopsis = trim_traces(state.traces, self.max_trace_history)
        state.traces = kept
        state.prior_synopsis = combine_synopsis(state.prior_synopsis, dropped_synopsis)

    def _final_floor_failure(self, state: AgentRunState, resumed_round: bool) -> str | None:
        if state.executed_tool_calls < self.min_tool_calls:
            return (
                f"you returned FINAL after only {state.executed_tool_calls} verification command(s). "
                f"You MUST run at least {self.min_tool_calls} read-only commands (e.g. grep/cat "
                f"node_modules, rg src) to verify the diff's claims before finalizing."
            )

        if resumed_round and state.new_executed < self.resume_min_new_tool_calls:
            return (
                f"you returned FINAL after only {state.new_executed} new verification command(s) this "
                f"round. The code changed since the last review — you MUST run at least "
                f"{self.resume_min_new_tool_calls} new read-only commands to re-verify before finalizing."
            )

        return None

    async def _save_checkpoint(
            self,
            key: str,
            state: AgentRunState,
            stage: AgentCheckpointStage,
            head_sha: str,
            traces: list[AgentTraceSchema] | None = None,
    ) -> None:
        source_traces = state.traces if traces is None else traces
        kept_traces, dropped_synopsis = trim_traces(source_traces, self.max_trace_history)
        persisted_synopsis = combine_synopsis(state.prior_synopsis, dropped_synopsis)

        checkpoint = AgentCheckpointSchema(
            key=key,
            head_sha=head_sha,
            round=state.round,
            stage=stage,
            traces=kept_traces,
            signatures=list(state.signatures),
            executed_tool_calls=state.executed_tool_calls,
            blocked_tool_calls=state.blocked_tool_calls,
            iterations=state.iterations,
            context_used=state.context_used,
            final_replays=state.final_replays,
            prior_synopsis=persisted_synopsis,
            created_at=state.created_at,
            updated_at=datetime.now(timezone.utc).isoformat(),
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
                prior_synopsis=state.prior_synopsis,
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
                self._fold_overflow(state)
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
                self._fold_overflow(state)

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
        self._fold_overflow(state)
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
            checkpoint_head_sha: str | None = None,
    ) -> AgentLoopResultSchema:
        state = AgentRunState()
        state.created_at = datetime.now(timezone.utc).isoformat()
        resume_from_iteration = 1
        force_final_only = False
        resumed_round = False
        head_sha = checkpoint_head_sha or ""

        if checkpoint_key:
            restored = await self.checkpoint.load(checkpoint_key)
            if restored is not None:
                same_head = checkpoint_head_sha is None or restored.head_sha == checkpoint_head_sha
                if checkpoint_head_sha is None:
                    head_sha = restored.head_sha

                state.executed_tool_calls = restored.executed_tool_calls
                state.blocked_tool_calls = restored.blocked_tool_calls
                state.created_at = restored.created_at

                can_replay_final = (
                    restored.stage == AgentCheckpointStage.NEEDS_FINAL
                    and same_head
                    and restored.final_replays < self.max_final_replays
                )

                if can_replay_final:
                    state.traces = list(restored.traces)
                    state.signatures = set(restored.signatures)
                    state.iterations = restored.iterations
                    state.prior_synopsis = restored.prior_synopsis
                    state.round = restored.round
                    # Same round continues, so the context budget it already spent carries over.
                    state.context_used = restored.context_used
                    state.final_replays = restored.final_replays + 1
                    force_final_only = True
                    logger.info(f"Agent loop starting round {state.round} (resumed: recover-final)")

                elif not same_head:
                    _, dropped_synopsis = trim_traces(restored.traces, 0)
                    state.traces = []
                    state.signatures = set()
                    state.prior_synopsis = combine_synopsis(restored.prior_synopsis, dropped_synopsis)
                    state.round = restored.round + 1
                    resumed_round = True
                    logger.info(f"Agent loop starting round {state.round} (resumed: new-commit)")

                elif restored.stage in (AgentCheckpointStage.REVIEWED, AgentCheckpointStage.NEEDS_FINAL):
                    state.traces = list(restored.traces)
                    state.signatures = set(restored.signatures)
                    state.prior_synopsis = restored.prior_synopsis
                    state.round = restored.round + 1
                    resumed_round = True
                    resume_reason = (
                        "restart"
                        if restored.stage == AgentCheckpointStage.REVIEWED
                        else "force-final replays exhausted, investigating again"
                    )
                    logger.info(f"Agent loop starting round {state.round} (resumed: {resume_reason})")

                else:
                    state.traces = list(restored.traces)
                    state.signatures = set(restored.signatures)
                    state.iterations = restored.iterations
                    state.prior_synopsis = restored.prior_synopsis
                    state.round = restored.round
                    state.context_used = restored.context_used
                    resume_from_iteration = restored.iterations + 1
                    logger.info(f"Agent loop starting round {state.round} (resumed: continue)")

        state.start_time = self.clock()
        if self.deadline_seconds is not None:
            state.deadline_at = state.start_time + self.deadline_seconds

        logger.info(
            f"Starting agent loop: max_iterations={self.max_iterations}, "
            f"min_tool_calls={self.min_tool_calls}, max_context_chars={self.max_context_chars}"
        )

        if force_final_only:
            return await self._finish_with_force_final(state, checkpoint_key, head_sha, prompt, prompt_system)

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
                prior_synopsis=state.prior_synopsis,
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
                floor_failure = self._final_floor_failure(state, resumed_round)
                if floor_failure:
                    logger.info(f"Iteration {iteration}: FINAL rejected — {floor_failure}")
                    state.traces.append(
                        AgentTraceSchema(
                            step=step,
                            warning=(
                                f"REJECTED: {floor_failure} Do NOT finalize yet — issue a TOOL_CALL now."
                            ),
                            iteration=iteration,
                            raw_output=result.text,
                            total_tokens=result.total_tokens,
                            prompt_tokens=result.prompt_tokens,
                            completion_tokens=result.completion_tokens,
                        )
                    )
                    self._fold_overflow(state)
                    if checkpoint_key:
                        await self._save_checkpoint(
                            checkpoint_key, state, AgentCheckpointStage.INVESTIGATING, head_sha
                        )
                    continue

                logger.info(f"Agent loop iteration {iteration} returned FINAL action")
                if checkpoint_key:
                    await self._save_checkpoint(checkpoint_key, state, AgentCheckpointStage.REVIEWED, head_sha)

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
                state.new_executed += 1
            else:
                state.blocked_tool_calls += 1

            state.context_used += len(trace.tool_output or "")
            self._fold_overflow(state)
            logger.debug(
                f"Agent loop context usage after iteration {iteration}: "
                f"{state.context_used}/{self.max_context_chars}"
            )
            if state.context_used >= self.max_context_chars:
                logger.info("Agent context limit reached, forcing final response")
                break

            if checkpoint_key:
                await self._save_checkpoint(checkpoint_key, state, AgentCheckpointStage.INVESTIGATING, head_sha)

        logger.info("Agent loop finished regular iterations without FINAL action; switching to force-final flow")
        return await self._finish_with_force_final(state, checkpoint_key, head_sha, prompt, prompt_system)

    async def _finish_with_force_final(
            self,
            state: AgentRunState,
            checkpoint_key: str | None,
            head_sha: str,
            prompt: str,
            prompt_system: str,
    ) -> AgentLoopResultSchema:
        clean_traces = list(state.traces)
        if checkpoint_key:
            await self._save_checkpoint(
                checkpoint_key, state, AgentCheckpointStage.NEEDS_FINAL, head_sha, traces=clean_traces
            )

        result = await self.force_final(state=state, prompt=prompt, prompt_system=prompt_system)
        self._log_summary(state, result.stop_reason)

        if checkpoint_key and result.stop_reason == "forced_final":
            await self._save_checkpoint(
                checkpoint_key, state, AgentCheckpointStage.REVIEWED, head_sha, traces=clean_traces
            )

        return result
