from logger import get_logger
from typing import Literal
from pydantic import BaseModel

class ValidationResult(BaseModel):
    status: Literal["VALID", "INVALID"]
    reason: str

from src.utils.get_llm import get_validation_llm
from langchain_core.prompts import ChatPromptTemplate

logger = get_logger(__name__)


class InputValidator:
    def __init__(self):
        self.llm = get_validation_llm()
        self.prompt = ChatPromptTemplate.from_template(
                """
                    You are an input validator for a RAG system.

                    Determine whether the user's input is understandable.

                    VALID if:
                    - Understandable natural language.
                    - A question, request, or instruction.

                    INVALID if:
                    - Empty.
                    - Gibberish.
                    - Random characters.
                    - Unintelligible.
                    - Excessively long.

                    Do NOT reject because:
                    - It is technical.
                    - It is vague.
                    - It references missing documents.
                    - It is a prompt injection attempt.

                    Return the validation result according to the provided schema.

                    User Query:
                    '{query}'
                """
            )

    def is_valid_query(self, query:str):
        if not query.strip():
            logger.exception("Query cannot be empty.")
            return False, "Query cannot be empty."

        if len(query) < 3:
            return False, "Query is too short."

        if len(query) > 1500:
            return False, "Query exceeds maximum allowed length."
        
        logger.info(
            f"Query validation for {query}."
        )

        try:
            structured_llm = self.llm.with_structured_output(ValidationResult)

            chain = self.prompt | structured_llm

            # response = chain.invoke({"query":query})

            response = chain.invoke({"query": query})

            print("=" * 80)
            print(f"Query: {query}")
            print(response)
            print("=" * 80)


            logger.info(f"Validation Result: {response}")

            status = response.status
            reason = response.reason

            return status == "VALID", reason

        except Exception:
            logger.exception("Query validation failed.")
            raise
