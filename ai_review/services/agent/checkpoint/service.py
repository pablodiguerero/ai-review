import hashlib
from pathlib import Path

import aiofiles

from ai_review.config import settings
from ai_review.libs.logger import get_logger
from ai_review.services.agent.checkpoint.schema import AgentCheckpointSchema
from ai_review.services.agent.checkpoint.types import AgentCheckpointServiceProtocol

logger = get_logger("AGENT_CHECKPOINT_SERVICE")


class AgentCheckpointService(AgentCheckpointServiceProtocol):
    def __init__(self):
        self.checkpoint_dir = settings.agent.checkpoint_dir

    def _path_for_key(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.checkpoint_dir / f"{digest}.json"

    async def load(self, key: str) -> AgentCheckpointSchema | None:
        if self.checkpoint_dir is None:
            return None

        path = self._path_for_key(key)
        if not path.exists():
            return None

        try:
            async with aiofiles.open(path, "r", encoding="utf-8") as aiofile:
                raw = await aiofile.read()

            return AgentCheckpointSchema.model_validate_json(raw)
        except Exception as error:
            logger.warning(f"Failed to load agent checkpoint {path}, ignoring it: {error}")
            return None

    async def save(self, key: str, checkpoint: AgentCheckpointSchema) -> None:
        if self.checkpoint_dir is None:
            return

        path = self._path_for_key(key)

        try:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

            async with aiofiles.open(path, "w", encoding="utf-8") as aiofile:
                await aiofile.write(checkpoint.model_dump_json(indent=2))

            logger.debug(f"Saved agent checkpoint {path}")
        except Exception as error:
            logger.warning(f"Failed to save agent checkpoint {path}: {error}")

    async def delete(self, key: str) -> None:
        if self.checkpoint_dir is None:
            return

        path = self._path_for_key(key)

        try:
            path.unlink(missing_ok=True)
            logger.debug(f"Deleted agent checkpoint {path}")
        except Exception as error:
            logger.warning(f"Failed to delete agent checkpoint {path}: {error}")
