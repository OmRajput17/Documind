from functools import lru_cache
from typing import List, Tuple
import time

import torch
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config import (
    RERANKER_MODEL_NAME,
    RERANK_TOP_K,
    MODEL_CACHE_PATH,
)

from logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=4)
def _load_cross_encoder(model_name: str) -> CrossEncoder:
    logger.info("Loading Reranker model: %s", model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Reranker device selected: %s", device)

    model = CrossEncoder(
        model_name_or_path=model_name,
        model_kwargs={
            "cache_dir": str(MODEL_CACHE_PATH),
        },
        device=device,
    )

    logger.info(
        "Reranker model loaded successfully on device: %s",
        device,
    )

    return model

class Reranker:
    """
    Cross-Encoder reranker for improving retrieval quality.
    """

    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        self.model = _load_cross_encoder(model_name)

    def rerank(
        self,
        query: str,
        results: List[Tuple[Document, float]],
        top_n: int = RERANK_TOP_K,
    ) -> List[Tuple[Document, float]]:
        """
        Rerank retrieved documents using a CrossEncoder.

        Args:
            query: User query.
            results: Retrieved (Document, score) tuples.
            top_n: Number of documents to return.

        Returns:
            List of reranked (Document, score) tuples.
        """

        if not results:
            logger.warning("No documents to rerank.")
            return []

        logger.info("Reranking %d retrieved chunks.", len(results))

        try:
            sentence_pairs = [
                (query, document.page_content)
                for document, _ in results
            ]

            # ---------------------------------------------------------- #
            # Benchmark: isolated inference timing                         #
            # ---------------------------------------------------------- #
            device = next(self.model.model.parameters()).device

            # Flush any pending GPU work so the timer starts clean
            if device.type == "cuda":
                torch.cuda.synchronize()

            t_start = time.perf_counter()

            scores = self.model.predict(
                sentence_pairs,
                batch_size=16,
                show_progress_bar=False,
                activation_fn=self.model.activation_fn,
            )

            score_values = [float(score) for score in scores]
            activation_name = type(self.model.activation_fn).__name__

            logger.info(
                "Reranker scores | labels=%s | activation=%s | scores=%s",
                self.model.num_labels,
                activation_name,
                [round(score, 6) for score in score_values],
            )

            # Synchronize again before stopping the clock so GPU latency
            # is included in the measurement
            if device.type == "cuda":
                torch.cuda.synchronize()

            inference_time = time.perf_counter() - t_start

            logger.info(
                "Reranker inference | device=%s | candidates=%d | batch_size=16 | time=%.4f s",
                device,
                len(sentence_pairs),
                inference_time,
            )
            # ---------------------------------------------------------- #

            reranked = sorted(
                (
                    (document, score)
                    for (document, _), score in zip(results, score_values)
                ),
                key=lambda x: x[1],
                reverse=True,
            )

            reranked = [
                (document, float(score))
                for document, score in reranked[:top_n]
            ]

            if reranked:
                logger.info(
                    "Successfully reranked top-%d chunks (best score: %.3f).",
                    len(reranked),
                    reranked[0][1],
                )
            else:
                logger.warning("Reranker returned no results.")

            return reranked

        except Exception:
            logger.exception("Reranking failed.")
            raise