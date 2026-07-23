# retrieval/reranker.py

from typing import List, Tuple
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from logger import get_logger

from config import RERANKER_MODEL_NAME

logger = get_logger(__name__)

class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        logger.info(f"Loading cross-encoder reranker: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(
            self,
            query: str,
            results: List[Tuple[Document, float]],
            top_n: int = 3
        ) -> List[Tuple[Document, float]]:
        """
            Re-scores retrieved chunks using a cross-encoder that directly
            compares query+chunk together (not pre-computed embedding similarity).

            Returns top_n reranked (Document, score) tuples.
        """

        if not results:
            logger.warning("No results to rerank.")
            return []

        docs = [doc for doc, _ in results]
        pairs = [(query, doc.page_content) for doc in docs]

        logger.info(
            f"Reranking {len(results)} retrieved chunks "
            f"for query: '{query}'"
        )

        scores = self.model.predict(
            pairs,
            batch_size=16,
            show_progress_bar=True
        )

        reranked = sorted(
            zip(docs, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        top_results = [
            (doc, float(score))
            for doc, score in reranked[:top_n]
        ]

        if top_results:
            logger.info(
                f"Reranked top score: {top_results[0][1]:.3f} "
                f"(was {results[0][1]:.3f} before reranking.)"
            )

        return top_results
        

