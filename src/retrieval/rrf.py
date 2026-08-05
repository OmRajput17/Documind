from collections import defaultdict
from typing import List, Tuple

from langchain_core.documents import Document

from logger import get_logger

logger = get_logger(__name__)

class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion (RRF) combines multiple ranked retrieval
    results into a single ranking.
    """

    def __init__(self, k: int = 60):
        self.k = k

    @staticmethod
    def _document_id(document: Document) -> str:
        """
        Generate a stable identifier for a document chunk.
        """

        return (
            f"{document.metadata.get('source','')}"
            f"::{document.metadata.get('page','')}"
            f"::{hash(document.page_content)}"
        )

    def fuse(self, *rankings: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """
        Fuse multiple ranked retrieval results.

        Args:
            *rankings:
                Multiple retrieval outputs.

        Returns:
            List[(Document, RRF Score)]
        """

        fused_scores = defaultdict(float)
        document_lookup: dict[str, Document] = {}

        for ranking in rankings:
            for rank, (document, _) in enumerate(ranking, start=1):
                doc_id = self._document_id(document=document)

                fused_scores[doc_id] += 1/(self.k + rank)

                document_lookup[doc_id] = document

        fused_results = sorted(
            fused_scores.items(),
            key=lambda x:x[1],
            reverse=True
        )

        logger.info(
            f"Fused {len(rankings)} rankings into "
            f"{len(fused_results)} unique chunks."
        )

        return [
            (document_lookup[doc_id], score)
            for doc_id, score in fused_results
        ]