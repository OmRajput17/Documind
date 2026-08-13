from typing import List, Tuple
from langchain_core.documents import Document
from logger import get_logger
from statistics import mean, median, stdev

from config import CONFIDENCE_THRESHOLD

logger = get_logger(__name__)


class ConfidenceEvaluator:
    """
    Evaluates retrieval quality using reranker score signals.

    Note:
        The resulting confidence value is a heuristic score,
        not a calibrated probability.
    """

    def __init__(self, top_k: int = 5, threshold: float = CONFIDENCE_THRESHOLD  ):
        self.top_k = top_k
        self.threshold = threshold

    def evaluate(self, results: List[Tuple[Document, float]])-> dict:
        """
        Evaluate confidence of reranked retrieval results.

        Args:
            results:
                Reranked documents with their scores.

        Returns:
            Dictionary containing confidence signals.
        """

        if not results:
            logger.warning("No retrieval results available.")
            return {
                "confidence": 0.0,
                "top_score": 0.0,
                "mean_top_k": 0.0,
                "median_score": 0.0,
                "score_std": 0.0,
                "score_gap": 0.0,
                "top_2_mean": 0.0,
                "top_3_mean": 0.0,
            }

        scores = [
            float(score)
            for _, score in results
        ]

        top_score = scores[0]

        k = min(self.top_k, len(scores))

        mean_top_k = mean(
            scores[:k]
        )

        median_score = median(scores)

        score_std = (
            stdev(scores)
            if len(scores) > 1
            else 0.0
        )

        score_gap = (
            top_score - scores[1]
            if len(scores) > 1
            else top_score
        )

        top_2_mean = mean(
            scores[:min(2, len(scores))]
        )  

        top_3_mean = mean(
            scores[:min(3, len(scores))]
        )



        # Initial heuristic.
        #
        # This is intentionally NOT treated as a probability.
        confidence = (
            0.6 * top_score + 0.4 * mean_top_k
        )

        metrices = {
            "confidence": round(confidence, 3),
            "top_score": round(top_score, 3),
            "mean_top_k": round(mean_top_k, 3),
            "median_score": round(median_score, 3),
            "score_std": round(score_std, 3),
            "score_gap": round(score_gap, 3),
            "top_2_mean": round(top_2_mean, 3),
            "top_3_mean": round(top_3_mean, 3),
        }

        logger.info(
            "Retrieval confidence: %.3f "
            "(top=%.3f, mean_top_k=%.3f, "
            "top_2=%.3f, top_3=%.3f, "
            "median=%.3f, std=%.3f, gap=%.3f)",
            confidence,
            top_score,
            mean_top_k,
            top_2_mean,
            top_3_mean,
            median_score,
            score_std,
            score_gap,
        )

        return metrices