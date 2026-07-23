from logger import get_logger
from typing import List

from langchain_core.prompts import ChatPromptTemplate

logger = get_logger(__name__)


class QueryRewriter:
    """
    Rewrites low-confidence user queries to improve document retrieval.
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template(
            """
           The following search query resulted in poor retrieval.

            Original query:
            {query}

            Previous reformulations:
            {previous_queries}

            Generate ONE new reformulation.

            Rules:
            - Preserve the original meaning.
            - Do not answer the question.
            - Do not repeat any previous reformulation.
            - Avoid Boolean operators (AND, OR, NOT).
            - Optimize for semantic vector search.
            - Return ONLY the rewritten query.
            """
        )

    def reformulate_query(self, original_query: str, previous_queries: List[str]) -> str:
        """
        Reformulate a query that produced low-confidence retrieval.

        Args:
            original_query: The user's original search query.

        Returns:
            A rewritten query optimized for semantic retrieval.
        """

        if not original_query.strip():
            raise ValueError("Query cannot be empty.")

        logger.info(
            "Reformulating query.\n"
            f"Original: {original_query}\n"
            f"Previous: {previous_queries}"
        )

        try:
            prompt = self.prompt.invoke({
                "query":original_query,
                "previous_queries":"\n".join(previous_queries)
            })
            response = self.llm.invoke(prompt)

            new_query = response.content.strip()

            if not new_query:
                logger.warning("LLM returned an empty query. Using original.")
                return original_query

            if new_query.lower() in {
                q.lower() for q in previous_queries
            }:
                logger.warning("Duplicate reformulation generated.")
                return original_query

            if new_query.lower() == original_query.lower():
                logger.info("Query unchanged after reformulation.")
                return original_query

            logger.info(f"Rewritten query: '{new_query}'")
            
            return new_query

        except Exception:
            logger.exception("Query Reformulation failed.")
            raise