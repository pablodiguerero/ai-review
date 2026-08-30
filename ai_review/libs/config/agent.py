import re
from pathlib import Path

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    enabled: bool = False
    max_iterations: int = Field(default=25, ge=1, le=100)
    min_tool_calls: int = Field(default=0, ge=0, le=50)
    allow_commands: list[re.Pattern[str]] = Field(
        default_factory=lambda: [
            re.compile(r"^ls(?:\s+.*)?$"),
            re.compile(r"^cat(?:\s+.*)?$"),
            re.compile(r"^head(?:\s+.*)?$"),
            re.compile(r"^tail(?:\s+.*)?$"),
            re.compile(r"^wc(?:\s+.*)?$"),
            re.compile(r"""^sed\s+-n\s+(['"]?)\d+(?:,\d+)?p\1\s+(?!-)\S+$"""),
            re.compile(
                r"^rg(?![\s\S]*(?:--pre\b|--pre-glob|--hostname-bin|--search-zip|\s-z(?:\s|$)))"
                r"(?:\s+[\s\S]*)?$"
            ),
            re.compile(r"^grep(?:\s+.*)?$"),
            re.compile(
                r"^find(?![\s\S]*(?:-exec|-execdir|-ok|-okdir|-delete|-fprint|-fprintf|-fls))(?:\s+[\s\S]*)?$"
            ),
            re.compile(
                r"^git(?![\s\S]*--output)\s+(?:status|show|diff|log|rev-parse|ls-files)(?:\s+[\s\S]*)?$"
            ),
        ]
    )
    command_timeout: int = Field(default=10, ge=1, le=120)
    max_total_context_chars: int = Field(default=40_000, ge=1_000, le=500_000)
    max_command_output_chars: int = Field(default=8_000, ge=1_000, le=500_000)
    empty_response_retries: int = Field(default=2, ge=0, le=10)
    force_final_attempts: int = Field(default=2, ge=1, le=10)
    # Reruns that may resume straight into force-final instead of investigating again. Replaying is
    # cheap when the previous job merely died mid-final; past this many tries the final itself is the
    # thing that is broken, and another replay would just reproduce it.
    max_final_replays: int = Field(default=1, ge=0, le=10)
    fallback_to_direct_chat: bool = False
    deadline_seconds: int | None = Field(default=None, ge=1)
    checkpoint_dir: Path | None = None
    resume_min_new_tool_calls: int = Field(default=2, ge=0, le=50)
    max_trace_history: int = Field(default=16, ge=1, le=200)
