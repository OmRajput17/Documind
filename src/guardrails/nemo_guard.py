from typing import Literal
from pydantic import BaseModel

import os

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
        
        os.environ["NVIDIA_API_KEY"] = NVIDIA_API_KEY

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

            # With enable_rails_exceptions=True, a blocked input comes back
            # as a structured exception object instead of a text refusal.
            if (
                isinstance(response, dict)
                and response.get("role") == "exception"
                and response.get("content", {}).get("type") == "InputRailException"
            ):
                return NemoGuardResult(
                    action="BLOCK",
                    reason="Blocked by NeMo Guardrails self_check_input."
                )

            return NemoGuardResult(
                action="ALLOW",
                reason="Passed NeMo Guardrails."
            )

        except Exception as e:
            logger.exception("NeMo Guardrails failed.")

            # Fail Open
            return NemoGuardResult(
                action="ALLOW",
                reason=f"Guard Failed: {str(e)}"  # fixed: was 'r', now 'e'
            )