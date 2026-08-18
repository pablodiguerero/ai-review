from ai_review.config import settings
from ai_review.libs.llm.output_json_parser import LLMOutputJSONParser
from ai_review.libs.logger import get_logger
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
    """The loop stopped before it verified anything, so there is no review to make."""


class AgentLoopService(AgentLoopServiceProtocol):
    def __init__(
            self,
            llm: LLMClientProtocol,
            prompt: PromptServiceProtocol,
            agent_tool: AgentToolServiceProtocol,
    ):
        self.llm = llm
        self.prompt = prompt
        self.agent_tool = agent_tool
        self.max_iterations = settings.agent.max_iterations
        self.max_context_chars = settings.agent.max_total_context_chars
        self.min_tool_calls = settings.agent.min_tool_calls
        self.empty_response_retries = settings.agent.empty_response_retries
        self.force_final_attempts = settings.agent.force_final_attempts

        self.parser = LLMOutputJSONParser(AgentStepSchema)
        self.traces: list[AgentTraceSchema] = []
        self.signatures: set[str] = set()
        self.context_used = 0

    def clear(self):
        self.traces = []
        self.signatures = set()
        self.context_used = 0
        logger.debug("Agent loop state cleared")

    async def _chat(self, prompt: str, prompt_system: str) -> ChatResultSchema:
        """LLM call that retries when the model returns empty content.

        Some providers (notably DeepSeek in JSON mode) occasionally return an
        empty body on HTTP 200; a retry almost always recovers it instead of
        aborting the whole review.
        """
        result = await self.llm.chat(prompt=prompt, prompt_system=prompt_system, json_mode=True)
        for attempt in range(1, self.empty_response_retries + 1):
            if (result.text or "").strip():
                return result
            logger.warning(f"LLM returned empty content; retrying ({attempt}/{self.empty_response_retries})")
            result = await self.llm.chat(prompt=prompt, prompt_system=prompt_system, json_mode=True)
        return result

    async def run_step(self, step: AgentStepSchema, chat: ChatResultSchema, iteration: int) -> AgentTraceSchema:
        if step.command in self.signatures:
            logger.debug(f"Duplicate tool call blocked at iteration {iteration}: {step.command}")
            return AgentTraceSchema(
                step=step,
                warning=f"Duplicate tool call blocked: {step.command}",
                iteration=iteration,
                raw_output=chat.text,
                total_tokens=chat.total_tokens,
                prompt_tokens=chat.prompt_tokens,
                completion_tokens=chat.completion_tokens,
            )

        self.signatures.add(step.command)
        logger.debug(f"Executing agent tool command at iteration {iteration}: {step.command}")
        tool_output = await self.agent_tool.execute(step.command)

        return AgentTraceSchema(
            step=step,
            iteration=iteration,
            raw_output=chat.text,
            tool_output=tool_output,
            total_tokens=chat.total_tokens,
            prompt_tokens=chat.prompt_tokens,
            completion_tokens=chat.completion_tokens,
        )

    async def force_final(
            self,
            prompt: str,
            prompt_system: str,
    ) -> AgentLoopResultSchema:
        logger.info("Forcing FINAL response after loop limits reached")

        agent_prompt_system = self.prompt.build_system_agent_request()
        last_result: ChatResultSchema | None = None

        for attempt in range(1, self.force_final_attempts + 1):
            agent_prompt = self.prompt.build_agent_request(
                traces=self.traces,
                force_final=True,
                original_prompt=prompt,
                original_prompt_system=prompt_system,
            )
            last_result = await self._chat(agent_prompt, agent_prompt_system)
            step = self.parser.parse_output(last_result.text)
            is_final = bool(step and step.action.is_final and (step.content or "").strip())
            logger.debug(
                f"Force-final attempt {attempt}/{self.force_final_attempts}; parsed_as_final={is_final}"
            )

            if is_final:
                self.traces.append(
                    AgentTraceSchema(
                        step=step,
                        warning="Forced final response after loop limits reached.",
                        iteration=len(self.traces) + 1,
                        raw_output=last_result.text,
                        total_tokens=last_result.total_tokens,
                        prompt_tokens=last_result.prompt_tokens,
                        completion_tokens=last_result.completion_tokens,
                    )
                )
                return AgentLoopResultSchema(
                    traces=self.traces,
                    final_text=step.content,
                    stop_reason="forced_final",
                )

            # Parsed but not a FINAL (e.g. another TOOL_CALL). Record the rejection
            # so the next attempt sees the feedback and finalizes instead of repeating.
            if step is not None:
                self.traces.append(
                    AgentTraceSchema(
                        step=step,
                        warning=(
                            "REJECTED: force-final requires a FINAL action carrying your review in "
                            "`content`. The loop is over — do NOT call tools now, return FINAL."
                        ),
                        iteration=len(self.traces) + 1,
                        raw_output=last_result.text,
                        total_tokens=last_result.total_tokens,
                        prompt_tokens=last_result.prompt_tokens,
                        completion_tokens=last_result.completion_tokens,
                    )
                )

        # No valid FINAL after every attempt. Never publish a raw or non-FINAL
        # step as the review — return empty text so the runner skips the comment
        # and the merge gate fail-closes, instead of posting a tool-call dump.
        logger.warning(
            f"Force-final produced no valid FINAL after {self.force_final_attempts} attempt(s); "
            f"skipping summary (no comment will be posted)"
        )
        self.traces.append(
            AgentTraceSchema(
                step=AgentStepSchema(
                    action=AgentAction.FINAL,
                    content="Agent could not produce a valid FINAL review after force-final.",
                ),
                warning="Force-final exhausted without a valid FINAL; no summary posted.",
                iteration=len(self.traces) + 1,
                raw_output=(last_result.text if last_result else ""),
                total_tokens=(last_result.total_tokens if last_result else None),
                prompt_tokens=(last_result.prompt_tokens if last_result else None),
                completion_tokens=(last_result.completion_tokens if last_result else None),
            )
        )
        return AgentLoopResultSchema(
            traces=self.traces,
            final_text="",
            stop_reason="forced_final_no_review",
        )

    async def run(self, prompt: str, prompt_system: str) -> AgentLoopResultSchema:
        self.clear()
        tool_calls = 0
        logger.info(
            f"Starting agent loop: max_iterations={self.max_iterations}, "
            f"min_tool_calls={self.min_tool_calls}, max_context_chars={self.max_context_chars}"
        )

        for iteration in range(1, self.max_iterations + 1):
            logger.debug(f"Agent loop iteration started: {iteration}")

            agent_prompt = self.prompt.build_agent_request(
                traces=self.traces,
                force_final=False,
                original_prompt=prompt,
                original_prompt_system=prompt_system,
            )
            agent_prompt_system = self.prompt.build_system_agent_request()
            logger.debug(
                f"Agent prompt for iteration {iteration} "
                f"(prompt_chars={len(agent_prompt)}, system_chars={len(agent_prompt_system)}, "
                f"traces={len(self.traces)})"
            )
            
            try:
                result = await self._chat(agent_prompt, agent_prompt_system)
            except Exception as error:
                if tool_calls < self.min_tool_calls:
                    logger.error(
                        f"Agent loop iteration {iteration} failed ({error}) after only "
                        f"{tool_calls}/{self.min_tool_calls} verification commands; aborting"
                    )
                    raise AgentVerificationAborted(
                        f"Agent aborted after {tool_calls}/{self.min_tool_calls} "
                        f"verification commands: {error}"
                    ) from error

                logger.warning(
                    f"Agent loop iteration {iteration} failed ({error}); "
                    f"switching to force-final flow with {len(self.traces)} traces kept"
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
                if tool_calls < self.min_tool_calls:
                    logger.info(
                        f"Iteration {iteration}: FINAL rejected — only {tool_calls}/{self.min_tool_calls} "
                        f"verification commands run; nudging the agent to keep verifying"
                    )
                    self.traces.append(
                        AgentTraceSchema(
                            step=step,
                            warning=(
                                f"REJECTED: you returned FINAL after only {tool_calls} verification "
                                f"command(s). You MUST run at least {self.min_tool_calls} read-only "
                                f"commands (e.g. grep/cat node_modules, rg src) to verify the diff's claims "
                                f"before finalizing. Do NOT finalize yet — issue a TOOL_CALL now."
                            ),
                            iteration=iteration,
                            raw_output=result.text,
                            total_tokens=result.total_tokens,
                            prompt_tokens=result.prompt_tokens,
                            completion_tokens=result.completion_tokens,
                        )
                    )
                    continue

                logger.info(f"Agent loop iteration {iteration} returned FINAL action")
                self.traces.append(
                    AgentTraceSchema(
                        step=step,
                        iteration=iteration,
                        raw_output=result.text,
                        total_tokens=result.total_tokens,
                        prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                    )
                )

                return AgentLoopResultSchema(
                    traces=self.traces,
                    final_text=step.content,
                    stop_reason="final",
                )

            trace = await self.run_step(step=step, chat=result, iteration=iteration)
            self.traces.append(trace)
            tool_calls += 1

            self.context_used += len(trace.tool_output or "")
            logger.debug(
                f"Agent loop context usage after iteration {iteration}: "
                f"{self.context_used}/{self.max_context_chars}"
            )
            if self.context_used >= self.max_context_chars:
                logger.info("Agent context limit reached, forcing final response")
                break

        logger.info("Agent loop finished regular iterations without FINAL action; switching to force-final flow")
        return await self.force_final(prompt=prompt, prompt_system=prompt_system)
