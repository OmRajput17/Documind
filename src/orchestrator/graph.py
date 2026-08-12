from typing import Literal

from langgraph.graph import StateGraph, START, END

from src.pipeline.state import RAGState

from logger import get_logger

logger = get_logger(__name__)

class RAGGraph:
    """
    LangGraph-based orchestration layer for DocuMind.

    Handles:
        - Input validation
        - PII masking
        - Prompt injection detection
        - NeMo Guardrails
        - Hybrid retrieval
        - Retrieval confidence evaluation
        - Query reformulation
        - Prompt construction
        - Generation
        - Response formatting
    """

    def __init__(
            self,
            validator,
            pii_masker,
            regex_guard,
            nemo_guard,
            retriever,
            confidence_evaluator,
            query_rewriter,
            prompt_builder,
            generator,
            formatter,
            max_reformulation: int = 1,
        ):
        self.validator = validator
        self.pii_masker = pii_masker
        self.regex_guard = regex_guard
        self.nemo_guard = nemo_guard

        self.retriever = retriever
        self.confidence_evaluator = self.confidence_evaluator
        self.query_rewriter = query_rewriter

        self.prompt_builder = prompt_builder
        self.generator = generator
        self.formatter = formatter


        self.max_reformulation = max_reformulation

        self.graph = self._build_graph()


    
    ## Nodes 

    def validate_input(self, state: RAGState):
        """Validate the original query."""

        query = state["query"]

        if not query.strip():
            return {
                "is_valid":False,
                "valition_reason":"Query cannot be empty",
                "status":"BLOCKED"
            }
        
        try:
            is_valid, reason = self.validator.is_valid_query(query)

            if not is_valid:
                logger.warning(
                    "Input Validation blocked query: %s",
                    reason
                )

            return {
                "is_valid": is_valid,
                "validation_reason": reason,
                "status": "VALIDATED" if is_valid else "BLOCKED",
            }

        except Exception as exc:
            logger.exception(
                "Input Validation failed."
            )

            return {
                "is_valid": False,
                "validation_reason": str(exc),
                "status": "ERROR",
                "error": str(exc),
            }
        
    
    def mask_pii(self, state:RAGState):
        """
        Detect and mask PII before downstream processing.
        """
        query = state["query"]

        result = self.pii_masker.check(query)

        entities = [
            entity.type
            for entity in result.entities
        ]

        if result.pii_detected:
            logger.info(
                "PII detected and masked: %s",
                entities,
            )

        return {
            "masked_query": result.masked_query,
            "pii_detected":result.pii_detected,
            "pii_entities":entities
        }

    
    def regex_guard_node(self, state: RAGState):
        """
        Run regex-based prompt injection detection.
        """
        query = state["masked_query"]

        result = self.regex_guard.check(query)

        blocked = result.action == "BLOCK"

        if blocked:
            logger.warning(
                "Regex prompt injection guard blocked query."
            )

        return {
            "regex_blocked": blocked,
            "regex_score": result.score,
            "status":"BLOCKED" if blocked else "GUARDED"
        }

    
