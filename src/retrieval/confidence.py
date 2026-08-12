from typing import List, Tuple
from langchain_core.documents import Document
from logger import get_logger

logger = get_logger(__name__)


class ConfidenceEvaluator:
    """
    Evaluates retrieval quality using reranker score signals.

    Note:
        The resulting confidence value is a heuristic score,
        not a calibrated probability.
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def evaluate(self, results: List[Tuple[Document, float]])-> dict:
        """
        Evaluate confidence of retrieved results.

        Returns:
            Dictionary containing retrieval confidence signals.
        """

        if not results:
            return {
                "confidence":0.0,
                "top_score":0.0,
                "mean_top_k":0.0,
                "score_gap":0.0,
            }

        scores = [
            float(score)
            for _, score in results
        ]

        top_score = scores[0]

        top_scores = scores[:self.top_k]

        mean_top_k = sum(top_scores) / len(top_scores)

        score_gap = (
            top_scores[0] - top_scores[-1]
            if len(top_scores) > 1
            else 0
        )

        # Initial heuristic.
        #
        # This is intentionally NOT treated as a probability.
        confidence = (
            0.6 * top_score + 0.4 * mean_top_k
        )

        results = {
            "confidence":confidence,
            "top_score":top_score,
            "mean_top_k":mean_top_k,
            "score_gap":score_gap,
        }

        logger.info(
            "Retrieval confidence: %.3f "
            "(top=%.3f, mean_top_k=%.3f, gap=%.3f)",
            result["confidence"],
            result["top_score"],
            result["mean_top_k"],
            result["score_gap"],
        )

        return results