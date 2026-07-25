import re
from typing import Dict, Any

from config import (
    PROMPT_INJECTION_PATTERNS,
    PROMPT_INJECTION_THRESHOLD,
)
from logger import get_logger

logger = get_logger(__name__)


class PromptInjectionGuard:
    """
    Detect prompt injection attempts using weighted regex matching.
    """

    def __init__(self):
        self.patterns = {
            re.compile(pattern, re.IGNORECASE): weight
            for pattern, weight in PROMPT_INJECTION_PATTERNS.items()
        }

    def check(self, query: str) -> Dict[str, Any]:
        """
        Check whether a user query contains prompt injection patterns.

        Args:
            query: User input query.

        Returns:
            Dictionary containing:
                blocked (bool)
                score (int)
                matched_patterns (list)
        """

        try:
            # Handle empty input
            if not query or not query.strip():
                return {
                    "blocked": False,
                    "score": 0,
                    "matched_patterns": [],
                }

            score = 0
            matched_patterns = []

            for pattern, weight in self.patterns.items():

                match = pattern.search(query)

                if match:
                    score += weight

                    matched_patterns.append(
                        {
                            "pattern": pattern.pattern,
                            "matched_text": match.group(),
                            "weight": weight,
                        }
                    )

            blocked = score >= PROMPT_INJECTION_THRESHOLD

            if blocked:
                logger.warning(
                    f"Prompt Injection Detected | Score={score} | Matches={matched_patterns}",
                )
            else:
                logger.info(
                    f"Prompt Injection Check Passed | Score={score}",
                )

            return {
                "blocked": blocked,
                "score": score,
                "matched_patterns": matched_patterns,
            }

        except Exception as e:
            logger.exception(
                "Unexpected error while checking prompt injection.",
            )

            # Fail open (recommended for this project)
            return {
                "blocked": False,
                "score": 0,
                "matched_patterns": [],
                "error": str(e),
            }