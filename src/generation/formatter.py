from typing import List, Tuple

from langchain_core.documents import Document

from src.datamodels.rag import RAGResponse, Source


class ResponseFormatter:
    """
    Formats the final response with answer and source metadata.
    """

    def format(self, answer: str, retrieved_docs: List[Tuple[Document, float]]) -> RAGResponse:
        """
        Format the generated answer with supporting sources.

        Args:
            answer: Generated answer.
            retrieved_docs: Retrieved documents with reranker scores.

        Returns:
            Formatted response.
        """

        

        return RAGResponse(
            answer=answer,
            sources=[
                Source(
                    source=doc.metadata.get("source", "Unknown"),
                    page=doc.metadata.get("page"),
                    relevance_score=round(score, 3),
                )
                for doc, score in retrieved_docs
            ],
        )