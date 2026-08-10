import re
import pickle
import hashlib
from pathlib import Path
from typing import List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from config import BM25_TOP_K, BM25_CACHE_PATH
from logger import get_logger

logger = get_logger(__name__)


def _corpus_fingerprint(documents: List[Document]) -> str:
    """
    Produce a stable, deterministic hash of the document corpus.

    Documents are sorted by (source, page) before hashing so that
    non-deterministic load ordering (e.g. from multithreaded loaders)
    does not cause a spurious fingerprint mismatch.

    Each document contributes its source path, page number, and page
    content to the hash.  Timestamps, memory addresses, and other
    runtime-volatile metadata are intentionally excluded.
    """
    # Sort by (source, page) so order is always the same regardless of
    # how the loader returned the documents.
    stable = sorted(
        documents,
        key=lambda d: (
            str(d.metadata.get("source", "")),
            int(d.metadata.get("page", 0)),
        ),
    )

    h = hashlib.md5()
    for doc in stable:
        h.update(str(doc.metadata.get("source", "")).encode())
        h.update(str(doc.metadata.get("page", "")).encode())
        h.update(doc.page_content.encode())

    return h.hexdigest()


class BM25Retriever:
    """
    Keyword-based retriever using the BM25 ranking algorithm.

    The tokenized index is persisted to disk (BM25_CACHE_PATH) so it
    does not have to be rebuilt from scratch on every startup.  A
    corpus fingerprint (MD5 of all page content) is stored alongside
    the index and used to detect stale caches automatically.
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

        fingerprint = _corpus_fingerprint(documents)
        self.bm25 = self._load_or_build(documents, fingerprint)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _load_or_build(
        self,
        documents: List[Document],
        fingerprint: str,
    ) -> BM25Okapi:
        cache_path = Path(BM25_CACHE_PATH)

        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)

                if cached.get("fingerprint") == fingerprint:
                    logger.info(
                        "BM25 cache hit — loading existing index from %s.", cache_path
                    )
                    self.tokenized_documents = cached["tokenized_documents"]
                    return cached["bm25"]

                logger.info(
                    "BM25 cache miss — corpus fingerprint changed. Rebuilding index."
                )
            except Exception:
                logger.warning(
                    "Failed to load BM25 cache — rebuilding index.",
                    exc_info=True,
                )

        return self._build_and_save(documents, fingerprint, cache_path)

    def _build_and_save(
        self,
        documents: List[Document],
        fingerprint: str,
        cache_path: Path,
    ) -> BM25Okapi:
        logger.info("Building BM25 index for %d documents.", len(documents))

        self.tokenized_documents = [
            self._tokenize(doc.page_content) for doc in documents
        ]
        bm25 = BM25Okapi(self.tokenized_documents)

        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(
                    {
                        "fingerprint": fingerprint,
                        "tokenized_documents": self.tokenized_documents,
                        "bm25": bm25,
                    },
                    f,
                )
            logger.info("BM25 index cached at %s.", cache_path)
        except Exception:
            logger.warning(
                "Could not persist BM25 cache — continuing without it.",
                exc_info=True,
            )

        logger.info("BM25 index built successfully.")
        return bm25

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text for BM25 indexing."""
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
            "BM25 retrieving top-%d chunks for query '%s'.", self.k, query
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
                    "BM25 retrieved %d chunks (best score: %.3f).",
                    len(results),
                    results[0][1],
                )
            else:
                logger.warning("BM25 found no relevant chunks.")

            return results

        except Exception:
            logger.exception("BM25 retrieval failed.")
            raise
