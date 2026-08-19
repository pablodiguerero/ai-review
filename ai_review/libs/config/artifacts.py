from pathlib import Path

from pydantic import BaseModel, Field


class ArtifactsConfig(BaseModel):
    llm_dir: Path = Field(default=Path("./artifacts/llm"))
    vcs_dir: Path = Field(default=Path("./artifacts/vcs"))
    llm_enabled: bool = False
    vcs_enabled: bool = False
