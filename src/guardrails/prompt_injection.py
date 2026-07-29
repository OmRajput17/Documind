import re
from config import (
    PROMPT_INJECTION_PATTERNS,
    PROMPT_INJECTION_BLOCK_THRESHOLD,
    PROMPT_INJECTION_REVIEW_THRESHOLD
)
from logger import get_logger

logger = get_logger(__name__)

from typing import Literal
from pydantic import BaseModel

class PromptInjectionResult(BaseModel):
    action: Literal["ALLOW", "REVIEW", "BLOCK"]
    score: int
    matched_patterns: list["MatchedPattern"]
    reason: str

class MatchedPattern(BaseModel):
    category: str
    pattern: str
    matched_text: str
    weight: int
    

class PromptInjectionGuard:
    """
    Detect prompt injection attempts using weighted regex matching.
    """

    def __init__(self):
        self.patterns = {
            re.compile(pattern, re.IGNORECASE): weight
            for pattern, weight in PROMPT_INJECTION_PATTERNS.items()
        }

    def check(self, query: str) -> PromptInjectionResult:
        """
        Check whether a user query contains prompt injection patterns.

        Args:
            query: User input query.

        Returns:
            Returns:
                PromptInjectionResult containing:
                    - action
                    - score
                    - matched_patterns
                    - reason
        """

        try:
            # Handle empty input
            if not query or not query.strip():
                return PromptInjectionResult(
                    action="ALLOW",
                    score=0,
                    matched_patterns=[],
                    reason="Empty query."
                )

            score = 0
            matched_patterns = []

            for pattern, weight in self.patterns.items():

                match = pattern.search(query)

                if match:
                    score += weight

                    matched_patterns.append(
                        MatchedPattern(
                            category="Unknown",   # We'll replace this later
                            pattern=pattern.pattern,
                            matched_text=match.group(),
                            weight=weight,
                        )
                    )

            if score >= PROMPT_INJECTION_BLOCK_THRESHOLD:
                action = "BLOCK"

            elif score >= PROMPT_INJECTION_REVIEW_THRESHOLD:
                action = "REVIEW"

            else:
                action = "ALLOW"

            if action == "BLOCK":
                logger.warning(
                    f"Prompt Injection Detected | Score={score} | Matches={matched_patterns}",
                )
            elif action == "REVIEW":
                logger.warning(
                    f"Prompt Injection Needs Review | Score={score} | Matches={matched_patterns}"
                )
            else:
                logger.info(
                    f"Prompt Injection Check Passed | Score={score}",
                )

            return PromptInjectionResult(
                action = action,
                score= score,
                matched_patterns=matched_patterns,
                reason= (
                    "Prompt Injection Detected."
                    if action == "BLOCK"
                    else "No Prompt Injection Detected."
                )
            )

        except Exception as e:
            logger.exception(
                "Unexpected error while checking prompt injection.",
            )

            # Fail open (recommended for this project)
            return PromptInjectionResult(
                action="ALLOW",
                score=0,
                matched_patterns=[],
                reason=f"Guard failed: {str(e)}"
            )