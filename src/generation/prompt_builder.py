# src/generation/prompt_builder.py

from typing import List, Tuple

from langchain_core.documents import Document

class PromptBuilder:
    """
    Builds the final prompt from the retrieved context and user query.
    """

    SYSTEM_TEMPLATE = """
        You are DocuMind, an AI assistant that answers questions strictly using the provided context.

        ### Instructions
        - Answer ONLY using the provided context.
        - Do NOT use your own knowledge.
        - If the answer cannot be found in the context, reply:
        "I couldn't find that information in the uploaded documents."
        - Do not fabricate, infer, or assume information.
        - Keep your answer clear, concise, and factual.
        - If multiple sources provide relevant information, combine them into a single coherent answer.

        ### Context
        {context}

        ### User Question
        {question}

        ### Answer
    """


    def build(self, query: str, retrieved_docs: List[Tuple[Document, float]])->str:
        """
        Build the final prompt.

        Args:
            query: User query.
            retrieved_docs : Reranked retrieval results.

        Returns:
            Prompt string.
        """

        context = "\n\n".join(
            f"[Source: {doc.metadata.get('source', 'Unknown')}, "
            f"Page: {doc.metadata.get('page', 'N/A')} "
            f"Relevance: {score:.3f}] \n"
            f"{doc.page_content}"
            for doc, score in retrieved_docs
        )

        return self.SYSTEM_TEMPLATE.format(
            context = context,
            question = query
        )