import warnings
import time
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_chroma import Chroma
from logger import get_logger

from src.retrieval.reranker import Reranker

from config import RETRIEVAL_TOP_K, RERANK_TOP_K

logger = get_logger(__name__)

class Retriever:
    def __init__(self, vectorstore: Chroma, k: int = RETRIEVAL_TOP_K):
        self.vectorstore = vectorstore
        self.k = k
        self.reranker = Reranker()
    

    def retrieve(self, query: str) -> List[Tuple[Document, float]]:
        """
        Retrieve the top-k most relevant document chunks along with
        their similarity scores.

        Args:
            query: User query.

        Returns:
            List of (Document, relevance_score) tuples.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        logger.info(f"Retrieving top-{self.k} chunks for query '{query}'")

        try:

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                results = self.vectorstore.similarity_search_with_relevance_scores(
                    query=query,
                    k=self.k
                )

            results = self.reranker.rerank(
                query=query,
                results=results,
                top_n=RERANK_TOP_K
            )

            if results:
                logger.info(
                    f"Retrieved and reranked {len(results)} chunks "
                    f"(best rerank score: {results[0][1]:.3f})"
                )
            else:
                logger.warning("No relevant chunks found.")

            return results

        except Exception:
            logger.exception("Retrieval Failed.")
            raise