from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document


class ResponseFormatter:
    """
    Formats the final response with answer and source metadata.
    """

    def format(self, answer: str, retrieved_docs: List[Tuple[Document, float]]) -> Dict[str, Any]:
        """
        Format the generated answer with supporting sources.

        Args:
            answer: Generated answer.
            retrieved_docs: Retrieved documents with reranker scores.

        Returns:
            Formatted response.
        """

        sources = [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "relevance_score": round(score, 3)
            }
            for doc, score in retrieved_docs
        ]

        return {
            "answer": answer,
            "sources": sources
        }