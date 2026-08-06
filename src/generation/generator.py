from langchain_core.messages import HumanMessage

from src.utils.get_llm import get_generation_llm

from logger import get_logger

logger = get_logger(__name__)

class Generator:
    """
    Generates answers using the configured LLM.
    """

    def __init__(self):
        logger.info(
            f"Loading generation model..."
        )

        self.llm = get_generation_llm()

        logger.info(
            f"Generation model loaded successfully."
        )

    def generate(self, prompt: str) -> str:
        """
        Generate an answer from the prompt.

        Args:
            prompt: Prompt built by PromptBuilder.

        Returns:
            Generated answer.
        """

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        logger.info("Generating Response.")

        try:

            response = self.llm.invoke(
                [
                    HumanMessage(content=prompt)
                ]
            )
                
            answer = response.content.strip()

            logger.info("Response Generated Successfully.")

            return answer

        except Exception:
            logger.exception(
                "generation failed."
            )
            raise