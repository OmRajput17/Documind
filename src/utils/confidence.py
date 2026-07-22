from typing import List, Tuple
from langchain_core.documents import Document
from logger import get_logger

from config import CONFIDENCE_THRESHOLD

logger = get_logger(__name__)


def check_confidence(results: List[Tuple[Document, float]]) -> tuple[bool, float]:
    """
    Returns (is_confident, top_score).
    Confidence = based on the TOP result's score — if even the best match
    is weak, the whole retrieval is unreliable.
    """

    if not results:
        logger.warning("No documents retrieved; confidence is zero.")
        return False, 0.0

    top_score = results[0][1]

    if not 0 <= top_score <= 1:
        logger.warning(
            f"Unexpected relevance score: {top_score}"
        )

    is_confident = top_score >= CONFIDENCE_THRESHOLD

    logger.info(
        f"Top score: {top_score:.3f} "
        f"(threshold={CONFIDENCE_THRESHOLD:.2f}) "
        f"-> confident={is_confident}"
    )
    return is_confident, top_score
