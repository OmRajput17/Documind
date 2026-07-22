from logger import get_logger

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
            The following search query resulted in low-confidence document retrieval.

            Rewrite it to maximize the likelihood of retrieving the most relevant
            document chunks.

            Do NOT answer the question.
            Return ONLY the rewritten search query.

            Original query:
            {query}

            Rewritten query:
            """
        )

    def reformulate_query(self, original_query: str) -> str:
        """
        Reformulate a query that produced low-confidence retrieval.

        Args:
            original_query: The user's original search query.

        Returns:
            A rewritten query optimized for semantic retrieval.
        """

        if not original_query.strip():
            raise ValueError("Query cannot be empty.")

        logger.info(f"Reformulating query: '{original_query}'")

        try:
            prompt = self.prompt.invoke({"query":original_query})
            response = self.llm.invoke(prompt)

            new_query = response.content.strip()

            if not new_query:
                logger.warning("LLM returned an empty query. Using original.")
                return original_query

            if new_query.lower() == original_query.lower():
                logger.info("Query unchanged after reformulation.")
                return original_query

            return new_query

        except Exception:
            logger.exception("Query Reformulation failed.")
            raise