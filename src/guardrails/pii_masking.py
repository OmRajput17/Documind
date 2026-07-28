import re

from pydantic import BaseModel

from config import (
    PII_PATTERNS,
    PII_PLACEHOLDERS,
)
from logger import get_logger

logger = get_logger(__name__)


class PIIEntity(BaseModel):
    type: str
    original: str
    masked: str


class PIIMaskingResult(BaseModel):
    masked_query: str
    pii_detected: bool
    entities: list[PIIEntity]


class PIIMaskingGuard:
    """
    Detects and masks Personally Identifiable Information (PII)
    from user queries before they are passed to downstream
    guardrails or retrieval components.
    """

    def __init__(self):
        self.patterns = {
            key: re.compile(pattern, re.IGNORECASE)
            for key, pattern in PII_PATTERNS.items()
        }

    def check(self, query: str) -> PIIMaskingResult:
        """
        Detect and mask PII from the input query.

        Args:
            query: User input.

        Returns:
            PIIMaskingResult containing:
                - masked_query
                - pii_detected
                - detected entities
        """

        if not query or not query.strip():
            return PIIMaskingResult(
                masked_query=query,
                pii_detected=False,
                entities=[]
            )

        masked_query = query
        entities: list[PIIEntity] = []

        for entity_type, pattern in self.patterns.items():

            placeholder = PII_PLACEHOLDERS.get(
                entity_type,
                f"[{entity_type}]"
            )

            for match in pattern.finditer(masked_query):

                entities.append(
                    PIIEntity(
                        type=entity_type,
                        original=match.group(),
                        masked=placeholder,
                    )
                )

            masked_query = pattern.sub(
                placeholder,
                masked_query,
            )

        if entities:

            detected_types = sorted(
                {entity.type for entity in entities}
            )

            logger.info(
                f"PII detected | Types={detected_types} | Count={len(entities)}"
            )

        else:

            logger.info("No PII detected.")

        return PIIMaskingResult(
            masked_query=masked_query,
            pii_detected=bool(entities),
            entities=entities,
        )