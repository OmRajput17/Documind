from typing import List

from langchain_core.prompts import ChatPromptTemplate

from logger import get_logger

logger = get_logger(__name__)


class QueryRewriter:
    """
    Rewrites low-confidence user queries to improve document retrieval.
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a query reformulation component for a Retrieval-Augmented
            Generation system.

            The following user query produced low-confidence document retrieval.

            Original query:
            {query}

            Previous reformulations:
            {previous_queries}

            Generate ONE new reformulation that is more likely to retrieve
            relevant documents.

            Rules:
            - Preserve the original meaning and intent.
            - Do not answer the question.
            - Do not introduce information that is not present in the original query.
            - Do not repeat any previous reformulation.
            - Avoid Boolean operators such as AND, OR, and NOT.
            - Make the query clear and retrieval-oriented.
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
            "Reformulating query. "
            "Original: '%s' | Previous: %s",
            original_query,
            previous_queries,
        )

        try:
            previous_queries_text = (
                "\n".join(
                    f"- {query}"
                    for query in previous_queries
                )
                if previous_queries
                else "None"
            )

            prompt = self.prompt.invoke(
                {
                    "query": original_query,
                    "previous_queries": previous_queries_text,
                }
            )

            response = self.llm.invoke(prompt)

            new_query = response.content.strip()


            if not new_query:
                logger.warning(
                    "LLM returned an empty reformulation. "
                    "Using original query."
                )
                return original_query
            
            new_query = new_query.strip("\"'")

            previous_normalized = {
                query.strip().lower()
                for query in previous_queries
            }

            if new_query.lower() in previous_normalized:
                logger.warning(
                    "Duplicate reformulation generated. "
                    "Using original query."
                )
                return original_query

            # --------------------------------------------------
            # Unchanged query check
            # --------------------------------------------------
            if new_query.lower() == original_query.strip().lower():
                logger.info(
                    "Query unchanged after reformulation. "
                    "Using original query."
                )
                return original_query

            logger.info(
                "Query successfully reformulated: '%s'",
                new_query,
            )

            return new_query

        except Exception:
            logger.exception(
                "Query reformulation failed."
            )
            raise