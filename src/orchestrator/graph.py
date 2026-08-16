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
            max_reformulations: int = 2,
        ):
        self.validator = validator
        self.pii_masker = pii_masker
        self.regex_guard = regex_guard
        self.nemo_guard = nemo_guard

        self.retriever = retriever
        self.confidence_evaluator = confidence_evaluator
        self.query_rewriter = query_rewriter

        self.prompt_builder = prompt_builder
        self.generator = generator
        self.formatter = formatter

        self.max_reformulations = max_reformulations

        self.graph = self._build_graph()


    #####################
    # Nodes 
    #####################

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

    async def nemo_guard_node(self, state: RAGState):
        """
        Run NeMoGuardrails
        """

        query = state["masked_query"]

        try:
            result = await self.nemo_guard.check(query)

            blocked = result.action == "BLOCK"

            if blocked:
                logger.warning(
                    "NeMo Guardrails Blocked query."
                )
            
            return {
                "nemo_blocked":blocked,
                "nemo_reason":result.reason,
                "status": "BLOCKED" if blocked else "GUARDED"
            }

        except Exception as exc:
            logger.exception(
                "NeMo Guardrails failed."
            )

            ## Preserve your existing fail-open behavior.
            return {
                "nemo_blocked":False,
                "nemo_reason":f"Guard Failed: {exc}",
                "status":"GUARDED"
            }
    
    def retrieve(self, state: RAGState):
        """
        Run hybrid retrieval using the current retrieval query.
        """

        query = state.get(
            "reformulated_query"
        ) or state["masked_query"]

        logger.info(
            "Running hybrid retrieval for query: '%s'",
            query,
        )

        results = self.retriever.retrieve(
            query = query
        )

        logger.info(
            "Retrieved %d chunks.",
            len(results),
        )

        return {
            "retrieved_docs":results
        }

    def evaluate_confidence(self, state:RAGState):
        """
        Evaluate the quality of retrieved documents.
        """

        results = state.get(
            "retrieved_docs",
            []
        )

        metrics = self.confidence_evaluator.evaluate(
            results
        )

        logger.info(
            "Retrieval confidence: %.3f",
            metrics["confidence"],
        )

        return {
            "retrieval_confidence": metrics["confidence"],
        }

    def rewrite_query(self, state:RAGState):
        """
        Reformulate the original query after low-confidence retrieval.
        """

        original_query = state["masked_query"]

        previous_queries = state.get(
            "rewrite_history",
            []
        )

        new_query = self.query_rewriter.reformulate_query(
            original_query=original_query,
            previous_queries=previous_queries
        )

        current_count = state.get(
            "reformulation_count",
            0
        )
        
        return {
            "reformulated_query":new_query,
            "reformulation_count":current_count+1,
            "rewrite_history":[
                *previous_queries,
                new_query,
            ],
        }

    def build_prompt(self, state: RAGState):
        """
        Build the generation prompt from retrieved context.
        """

        query = (
            state.get("best_query")
            or state["masked_query"]
        )

        prompt = self.prompt_builder.build(
            query=query,
            retrieved_docs = state["best_retrieved_docs"]
        )

        return {
            "prompt":prompt
        }
    
    def generate(self, state:RAGState):
        """
        Generate the final answer.
        """
        answer = self.generator.generate(
            prompt = state["prompt"]
        )

        return {
            "answer":answer
        }    

    def format_response(self, state:RAGState):
        """
        Format the final RAG response
        """

        response = self.formatter.format(
            answer = state["answer"],
            retrieved_docs = state["best_retrieved_docs"]
        )

        return {
            "response":response,
            "status":"COMPLETED"
        }

    def update_best_retrieval(self, state: RAGState):
        """
        Keep the highest-confidence retrieval result encountered
        during the reformulation loop.
        """

        current_confidence = state.get(
            "retrieval_confidence",
            0.0,
        )

        current_docs = state.get(
            "retrieved_docs",
            [],
        )

        current_query = (
            state.get("reformulated_query")
            or state["masked_query"]
        )

        best_confidence = state.get(
            "best_retrieval_confidence"
        )

        # First retrieval
        if best_confidence is None:
            logger.info(
                "Initializing best retrieval. "
                "Confidence: %.3f",
                current_confidence,
            )

            return {
                "best_retrieved_docs": current_docs,
                "best_retrieval_confidence": current_confidence,
                "best_query": current_query,
            }

        # New best
        if current_confidence > best_confidence:
            logger.info(
                "New best retrieval found. "
                "Previous: %.3f | Current: %.3f",
                best_confidence,
                current_confidence,
            )

            return {
                "best_retrieved_docs": current_docs,
                "best_retrieval_confidence": current_confidence,
                "best_query": current_query,
            }

        # Keep existing best
        logger.info(
            "Keeping previous best retrieval. "
            "Best: %.3f | Current: %.3f",
            best_confidence,
            current_confidence,
        )

        return {}


    #####################
    # Conditional Routing
    #####################

    @staticmethod
    def route_validation(state: RAGState) -> Literal["mask_pii", "end"]:
        """
        Route based on Input Validation
        """
        if state.get("is_valid", False):
            return "mask_pii"

        return "end"

    @staticmethod
    def route_regex(state:RAGState)->Literal["nemo_guard","retrieve","end"]:
        """
        Route based on regex prompt-injection detection.

        Current behavior:
            BLOCK → END
            ALLOW → NeMo Guardrails

        We can introduce risk-based NeMo skipping later
        after measuring its behavior.
        """
        if state.get("regex_blocked", False):
            return "end"
        
        return "nemo_guard"

    @staticmethod
    def route_nemo(state:RAGState)->Literal["retrieve", "end"]:
        """
        Route based on NeMo Guardrails.
        """
        if state.get("nemo_blocked", False):
            return "end"

        return "retrieve"

    def route_confidence(self, state:RAGState)->Literal["rewrite_query", "build_prompt"]:
        """
        Decide whether retrieval is good enough.

        If the maximum number of reformulations has already
        been reached, continue with the best available results.
        """
        reformulation_count= state.get(
            "reformulation_count",
            0
        )

        current_confidence = state.get(
            "retrieval_confidence",
            0.0
        )

        best_confidence = state.get(
            "best_retrieval_confidence",
            0.0
        )

        threshold = self.confidence_evaluator.threshold

        if current_confidence >= threshold:
            logger.info(
                "Current Retrieval confidence %.3f >= %.3f. "
                "Continuing to generation.",
                current_confidence,
                threshold,
            )

            return "build_prompt"
        
        if reformulation_count >= self.max_reformulations:
            logger.warning(
                "Maximum reformulations reached. "
                "Current confidence: %.3f | "
                "Best confidence: %.3f | "
                "Threshold: %.3f. "
                "Continuing with best retrieval results.",
                current_confidence,
                best_confidence,
                threshold,
            )

            return "build_prompt"
        
        logger.info(
            "Current retrieval confidence %.3f < %.3f. "
            "Reformulating query. "
            "Best confidence so far: %.3f.",
            current_confidence,
            threshold,
            best_confidence,
        )

        return "rewrite_query"

    ###############################
    # Graph Construction
    ###############################

    def _build_graph(self):

        workflow = StateGraph(RAGState)

        # --------------------------------------------------
        # Nodes
        # --------------------------------------------------

        workflow.add_node(
            "validate_input",
            self.validate_input,
        )

        workflow.add_node(
            "mask_pii",
            self.mask_pii,
        )

        workflow.add_node(
            "regex_guard",
            self.regex_guard_node,
        )

        workflow.add_node(
            "nemo_guard",
            self.nemo_guard_node,
        )

        workflow.add_node(
            "retrieve",
            self.retrieve,
        )

        workflow.add_node(
            "evaluate_confidence",
            self.evaluate_confidence,
        )

        workflow.add_node(
            "update_best_retrieval",
            self.update_best_retrieval,
        )

        workflow.add_node(
            "rewrite_query",
            self.rewrite_query,
        )

        workflow.add_node(
            "build_prompt",
            self.build_prompt,
        )

        workflow.add_node(
            "generation",
            self.generate,
        )

        workflow.add_node(
            "format_response",
            self.format_response,
        )

        # --------------------------------------------------
        # Start
        # --------------------------------------------------

        workflow.add_edge(
            START,
            "validate_input",
        )

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        workflow.add_conditional_edges(
            "validate_input",
            self.route_validation,
            {
                "mask_pii": "mask_pii",
                "end": END,
            },
        )

        # --------------------------------------------------
        # Input Guardrails
        # --------------------------------------------------

        workflow.add_edge(
            "mask_pii",
            "regex_guard",
        )

        workflow.add_conditional_edges(
            "regex_guard",
            self.route_regex,
            {
                "nemo_guard": "nemo_guard",
                "retrieve": "retrieve",
                "end": END,
            },
        )

        workflow.add_conditional_edges(
            "nemo_guard",
            self.route_nemo,
            {
                "retrieve": "retrieve",
                "end": END,
            },
        )

        # --------------------------------------------------
        # Retrieval / Confidence / Reformulation Loop
        # --------------------------------------------------

        workflow.add_edge(
            "retrieve",
            "evaluate_confidence",
        )

        workflow.add_edge(
            "evaluate_confidence",
            "update_best_retrieval",
        )

        workflow.add_conditional_edges(
            "update_best_retrieval",
            self.route_confidence,
            {
                "rewrite_query": "rewrite_query",
                "build_prompt": "build_prompt",
            },
        )

        workflow.add_edge(
            "rewrite_query",
            "retrieve",
        )

        # --------------------------------------------------
        # Linear Generation Pipeline
        # --------------------------------------------------

        workflow.add_edge(
            "build_prompt",
            "generation",
        )

        workflow.add_edge(
            "generation",
            "format_response",
        )

        workflow.add_edge(
            "format_response",
            END,
        )

        return workflow.compile()

    ##########################
    # Public API
    ##########################

    async def invoke(self, query: str, config: dict | None = None) -> RAGState:
        """
        Execute the DocuMind LangGraph.

        Args:
            query: User query.
            config: Optional LangGraph RunnableConfig (LangSmith tags, metadata, etc.)

        Returns:
            Final graph state.
        """

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        initial_state: RAGState = {
            "query": query,
            "masked_query": "",
            "reformulation_count": 0,
            "max_reformulations": self.max_reformulations,
            "rewrite_history": [],
            "status": "STARTED"
        }

        logger.info(
            "Starting DocuMind LangGraph"
        )

        try:
            result = await self.graph.ainvoke(
                initial_state,
                config=config,
            )

            logger.info(
                "Documind LangGraph Complete"
            )

            return result
        except Exception:
            logger.exception(
                "DocuMind LangGraph execution failed."
            )
            raise
