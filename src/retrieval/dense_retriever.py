import warnings
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import DENSE_TOP_K
from logger import get_logger

logger = get_logger(__name__)

class DenseRetriever:
    def __init__(self, vectorstore: Chroma, k: int = DENSE_TOP_K):
        self.vectorstore = vectorstore
        self.k = k

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
                    k=self.k,
                )

            if results:
                logger.info(f"Retrieved {len(results)} chunks")
            else:
                logger.warning("No relevant chunks found.")

            return results

        except Exception:
            logger.exception("Retrieval Failed.")
            raise
