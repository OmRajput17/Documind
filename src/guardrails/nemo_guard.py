from typing import Literal
from pydantic import BaseModel

from logger import get_logger

from nemoguardrails import LLMRails, RailsConfig
from config import NVIDIA_API_KEY

logger = get_logger(__name__)

class NemoGuardResult(BaseModel):
    action: Literal["ALLOW", "BLOCK"]
    reason: str

    

class NemoGuard:

    def __init__(self):

        if not NVIDIA_API_KEY:
            raise ValueError(
                "NVIDIA_API_KEY not found."
            )

        self.config = RailsConfig.from_path("src/guardrails/nemo")
        self.rails = LLMRails(self.config)

    
    async def check(self, query: str) -> NemoGuardResult:
        """
        Run NeMo Guardrails semantic prompt injection detection.

        Args:
            query: User query.

        Returns:
            NemoGuardResult
        """

        try:
            logger.info(
                f"Running NeMo Guardrails for query: {query}."
            )
            
            response = await self.rails.generate_async(
                messages=[
                    {
                        "role":"user",
                        "content":query
                    }
                ]
            )

            answer = response["content"].strip().lower()

            logger.info(
                f"Nemo Response: {answer}"
            )

            if answer.startswith("yes"):
                return NemoGuardResult(
                    action = "BLOCK",
                    reason = "Blocked by NeMo Guardrails."
                )

            return NemoGuardResult(
                action = "ALLOW",
                reason = "Passed NeMo Guardrails."
            ) 

        except Exception as e:
            logger.exception(
                "NeMo Guardrails failed."
            )

            # Fail Open

            return NemoGuardResult(
                action = "ALLOW",
                reason = f"Guard Failed: {str(r)}"
            )