from typing import List, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config import (
    RERANKER_MODEL_NAME,
    RERANK_TOP_K,
    MODEL_CACHE_PATH
)

from logger import get_logger

logger = get_logger(__name__)

class Reranker:
    """
    Cross-Encoder reranker for improving retrieval quality.
    """

    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        logger.info(
            f"Loading Reranker model: {model_name}"
        )

        self.model = CrossEncoder(
            model_name_or_path=model_name,
            model_kwargs={"cache_dir": str(MODEL_CACHE_PATH)},
        )

        logger.info("Reranker loaded Successfully.")


    def rerank(
            self, 
            query: str, 
            results: List[Tuple[Document, float]], 
            top_n : int=RERANK_TOP_K
        )-> List[Tuple[Document, float]]:
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

        logger.info(
            f"Reranking {len(results)} retrieved chunks."
        )

        try:

            sentence_pairs = [
                (query, document.page_content)
                for document, _ in results
            ]

            scores = self.model.predict(
                sentence_pairs,
                batch_size=16,
                show_progress_bar=False,
            )

            reranked = sorted(
                (
                    (document, score)
                    for (document, _), score in zip(results, scores)
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
                    f"Successfully reranked top-{len(reranked)} chunks "
                    f"(best score: {reranked[0][1]:.3f})"
                )
            else:
                logger.warning("Reranker returned no results.")

            return reranked

        except Exception:
            logger.exception(
                "Reranking failed."
            )
            raise