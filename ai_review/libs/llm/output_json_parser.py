import re
from typing import TypeVar, Generic, Type

from pydantic import BaseModel, ValidationError

from ai_review.libs.json import sanitize_json_string
from ai_review.libs.logger import get_logger

logger = get_logger("LLM_JSON_PARSER")

T = TypeVar("T", bound=BaseModel)

CLEAN_JSON_BLOCK_RE = re.compile(r"```(?:json)?(.*?)```", re.DOTALL | re.IGNORECASE)


class LLMOutputJSONParser(Generic[T]):
    """Reusable JSON parser for LLM responses."""

    def __init__(self, model: Type[T]):
        self.model = model
        self.model_name = self.model.__name__

    def try_parse(self, raw: str) -> T | None:
        logger.debug(f"[{self.model_name}] Attempting JSON parse (len={len(raw)})")

        try:
            return self.model.model_validate_json(raw)
        except ValidationError as error:
            logger.warning(f"[{self.model_name}] Raw JSON parse failed: {error}")
            cleaned = sanitize_json_string(raw)

            if cleaned != raw:
                logger.debug(f"[{self.model_name}] Sanitized JSON differs, retrying parse...")
                try:
                    return self.model.model_validate_json(cleaned)
                except ValidationError as error:
                    logger.warning(f"[{self.model_name}] Sanitized JSON still invalid: {error}")
                    return None
            else:
                logger.debug(f"[{self.model_name}] Sanitized JSON identical — skipping retry")
                return None

    def parse_output(self, output: str) -> T | None:
        output = (output or "").strip()
        if not output:
            logger.warning(f"[{self.model_name}] Empty LLM output")
            return None

        logger.debug(f"[{self.model_name}] Parsing output (len={len(output)})")

        if match := CLEAN_JSON_BLOCK_RE.search(output):
            logger.debug(f"[{self.model_name}] Found fenced JSON block, extracting...")
            output = match.group(1).strip()

        if parsed := self.try_parse(output):
            logger.info(f"[{self.model_name}] Successfully parsed")
            return parsed

        # Salvage: the model sometimes emits a valid JSON object and then keeps
        # generating trailing text (e.g. a fabricated "Tool output: ..." after a
        # ReAct step). That makes a whole-string parse fail on trailing chars.
        # Parse just the first balanced object and ignore everything after it.
        candidate = self._extract_first_json_object(output)
        if candidate and candidate != output:
            logger.debug(f"[{self.model_name}] Retrying parse on the first JSON object (trailing text ignored)")
            if parsed := self.try_parse(candidate):
                logger.info(f"[{self.model_name}] Successfully parsed first JSON object")
                return parsed

        logger.error(f"[{self.model_name}] No valid JSON found in output")
        return None

    @staticmethod
    def _extract_first_json_object(text: str) -> str | None:
        """Return the substring of the first balanced top-level ``{...}`` object, or None.

        String literals (and their escapes) are skipped so braces inside strings
        do not affect nesting depth.
        """
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]

        return None
