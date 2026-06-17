import re

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    enabled: bool = False
    max_iterations: int = Field(default=25, ge=1, le=100)
    min_tool_calls: int = Field(default=0, ge=0, le=50)
    allow_commands: list[re.Pattern[str]] = Field(
        default_factory=lambda: [
            re.compile(r"^ls(?:\s+.*)?$"),
            re.compile(r"^cat(?:\s+.*)?$"),
            re.compile(r"^rg(?:\s+.*)?$"),
            re.compile(r"^grep(?:\s+.*)?$"),
            re.compile(r"^git\s+(?:status|show|diff|log|rev-parse|ls-files)(?:\s+.*)?$"),
        ]
    )
    command_timeout: int = Field(default=10, ge=1, le=120)
    max_total_context_chars: int = Field(default=40_000, ge=1_000, le=500_000)
    max_command_output_chars: int = Field(default=40_000, ge=1_000, le=500_000)
    # Extra LLM attempts when a step comes back with empty content. Some providers
    # (e.g. DeepSeek JSON mode) occasionally return an empty body; one retry usually
    # recovers it instead of aborting the whole review.
    empty_response_retries: int = Field(default=2, ge=0, le=10)
    # Attempts to coax a valid FINAL out of the force-final flow. If none of them
    # parse as a real FINAL, the loop posts nothing rather than dumping a raw step.
    force_final_attempts: int = Field(default=2, ge=1, le=10)
