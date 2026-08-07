import re
from typing import List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from config import BM25_TOP_K
from logger import get_logger

logger = get_logger(__name__)


class BM25Retriever:
    """
    Keyword-based retriever using the BM25 ranking algorithm.
    """

    def __init__(self, documents: List[Document], k: int = BM25_TOP_K):
        """
        Initialize the BM25 retriever.

        Args:
            documents: List of document chunks.
            k: Number of documents to retrieve.
        """

        if not documents:
            raise ValueError("Document List cannot be empty.")

        self.documents = documents
        self.k = k

        logger.info(f"Building BM25 index for {len(documents)} documents.")

        self.tokenized_documents = [
            self._tokenize(doc.page_content)
            for doc in documents
        ]

        self.bm25 = BM25Okapi(self.tokenized_documents)

        logger.info("BM25 index created successfully.")

    
    @staticmethod
    def _tokenize(text: str)-> List[str]:
        """
        Tokenize text for BM25 indexing.
        """

        return re.findall(r"\w+", text.lower())

    def retrieve(self, query: str) -> List[Tuple[Document, float]]:
        """
        Retrieve the top-k documents using BM25.

        Args:
            query: User query.

        Returns:
            List of (Document, BM25 score) tuples.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        logger.info(
            f"BM25 retrieving top-{self.k} chunks for query '{query}'."
        )

        try:
            tokenized_query = self._tokenize(query)
            
            scores = self.bm25.get_scores(tokenized_query)

            top_indices = np.argsort(scores)[::-1][: self.k]

            results = [
                (self.documents[idx], float(scores[idx]))
                for idx in top_indices
                if scores[idx] > 0
            ]

            if results:
                logger.info(
                    f"BM25 retrieved {len(results)} chunks "
                    f"(best score: {results[0][1]:.3f})"
                )
            else:
                logger.warning("BM25 found no relevant chunks.")

            return results

        except Exception:
            logger.exception("BM25 retrieval failed.")
            raise

