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
            Generation (RAG) system.

            The following user query produced low-confidence document retrieval.

            Original query:
            {query}

            Previous reformulations:
            {previous_queries}

            Your task is to generate ONE improved search query that is more likely
            to retrieve the information needed to answer the original question.

            Rules:
            - Identify the user's primary information need.
            - Preserve the original meaning and intent.
            - Rewrite the query for document retrieval, not for conversation.
            - Focus on the important concepts, entities, terminology, and relationships
            required to retrieve the relevant information.
            - Remove irrelevant conversational details, greetings, personal information,
            identifiers, and other details that do not help retrieve the answer.
            - Ignore PII placeholders such as [AADHAAR], [EMAIL], [PHONE], [PAN], etc.
            unless the PII itself is directly relevant to the user's question.
            - Do not answer the question.
            - Do not introduce facts, entities, concepts, or assumptions that are not
            present or clearly implied by the original query.
            - Do not change the subject or scope of the original question.
            - Do not repeat any previous reformulation.
            - Avoid Boolean operators such as AND, OR, and NOT.
            - Prefer a concise, precise natural-language query suitable for semantic
            vector search.
            - Return ONLY the rewritten query.
            - Do not include explanations, labels, quotes, or reasoning.

            Example 1:

            Original:
            My Aadhaar number is [AADHAAR]. Explain positional encoding.

            Reformulation:
            Explain positional encoding in Transformer architectures.

            Example 2:

            Original:
            Can you tell me what self attention actually does in transformers?

            Reformulation:
            Explain the role of self-attention in Transformer architectures.

            Example 3:

            Original:
            I uploaded some documents about BERT. What does masked language
            modeling mean?

            Reformulation:
            Explain masked language modeling in BERT.

            Example 4:

            Original:
            What is the difference between supervised and unsupervised learning
            in machine learning?

            Reformulation:
            Compare supervised and unsupervised machine learning.

            Generate ONE improved retrieval query now.
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