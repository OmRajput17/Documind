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
    You are a query rewriter for a technical document search engine.

    TASK:
    Rewrite the user query into a short, precise search query optimized
    for retrieving relevant technical documents.

    USER QUERY:
    {query}

    PREVIOUS RETRIEVAL QUERIES:
    {previous_queries}

    RULES:
    - Preserve the exact meaning and scope of the user query.
    - Do not answer the query.
    - Identify the central technical concept and the specific mechanism,
      component, process, or relationship being asked about.
    - Replace conversational phrasing with precise technical terminology
      likely to appear in technical documents.
    - Do not add facts, concepts, entities, or assumptions not implied
      by the original query.
    - Keep the rewritten query concise, typically 4–10 terms.
    - Do not use AND, OR, NOT, or quotation marks.
    - Do not use complete sentences.
    - If previous retrieval queries are provided, do not merely reorder
      or slightly modify their words.
    - Use a genuinely different retrieval perspective when possible:
      mechanism → components
      concept → technical terminology
      purpose → implementation
      process → computation
      relationship → interaction
    - Avoid repeating the dominant terminology of previous queries when
      an equivalent technical alternative exists.
    - Output exactly ONE line containing only the rewritten query.
    - No labels, explanations, punctuation, or additional text.

    EXAMPLES:

    Query: What information do Transformers use to determine the order
    of tokens in an input sequence?
    Output: Transformer positional encoding token position sequence order

    Query: How does a Transformer know which words should pay attention
    to each other?
    Output: Transformer self-attention relationships between tokens attention weights

    Query: How does a model remember where each word occurs in a sentence?
    Output: Transformer positional encoding word position sequence

    Now rewrite the USER QUERY above.

    Output:
    """
)

    def reformulate_query(self, original_query: str, previous_queries: List[str]) -> str:
        """
        Reformulate a query that produced low-confidence retrieval.

        Args:
            original_query:
                The original sanitized user query.

            previous_queries:
                Previously generated reformulations used to avoid
                repeating the same retrieval query.

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

            if new_query.strip().lower() in previous_normalized:
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