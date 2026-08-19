from ai_review.config import settings
from ai_review.libs.logger import get_logger
from ai_review.services.agent.loop.service import AgentVerificationAborted
from ai_review.services.agent.loop.types import AgentLoopServiceProtocol
from ai_review.services.artifacts.schema.llm import LLMArtifactAgentSchema
from ai_review.services.artifacts.types import ArtifactsServiceProtocol
from ai_review.services.cost.schema import CalculateCostSchema
from ai_review.services.cost.types import CostServiceProtocol
from ai_review.services.hook import hook
from ai_review.services.llm.types import LLMClientProtocol
from ai_review.services.review.gateway.types import ReviewLLMGatewayProtocol

logger = get_logger("REVIEW_AGENT_LLM_GATEWAY")


class ReviewAgentLLMGateway(ReviewLLMGatewayProtocol):
    def __init__(
            self,
            llm: LLMClientProtocol,
            cost: CostServiceProtocol,
            artifacts: ArtifactsServiceProtocol,
            agent_loop: AgentLoopServiceProtocol,
            fallback_gateway: ReviewLLMGatewayProtocol,
    ):
        self.llm = llm
        self.cost = cost
        self.artifacts = artifacts
        self.agent_loop = agent_loop
        self.fallback_gateway = fallback_gateway

    async def ask(self, prompt: str, prompt_system: str, checkpoint_key: str | None = None) -> str:
        try:
            await hook.emit_chat_start(prompt, prompt_system)
            loop_result = await self.agent_loop.run(
                prompt=prompt,
                prompt_system=prompt_system,
                checkpoint_key=checkpoint_key,
            )

            report = self.cost.calculate(
                CalculateCostSchema(
                    prompt_tokens=loop_result.prompt_tokens,
                    completion_tokens=loop_result.completion_tokens
                )
            )
            if report:
                logger.info(report.pretty())

            await hook.emit_chat_complete(loop_result.final_text, report)
            await self.artifacts.save_llm(
                prompt=prompt,
                response=loop_result.final_text,
                cost_report=report,
                prompt_system=prompt_system,
                agent=LLMArtifactAgentSchema(
                    iterations=loop_result.iterations,
                    executed_tool_calls=loop_result.executed_tool_calls,
                    blocked_tool_calls=loop_result.blocked_tool_calls,
                    stop_reason=loop_result.stop_reason,
                ),
            )
            return loop_result.final_text
        except AgentVerificationAborted as error:
            logger.error(f"Agent mode aborted before verifying anything, posting nothing: {error}")
            await hook.emit_chat_error(prompt, prompt_system)
            return ""
        except Exception as error:
            if not settings.agent.fallback_to_direct_chat:
                logger.error(f"Agent mode failed, posting nothing: {error}")
                await hook.emit_chat_error(prompt, prompt_system)
                return ""

            logger.exception(f"Agent mode failed, falling back to direct chat: {error}")
            await hook.emit_chat_error(prompt, prompt_system)
            return await self.fallback_gateway.ask(prompt, prompt_system, checkpoint_key=checkpoint_key)
