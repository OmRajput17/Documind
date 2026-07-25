from typing import List, Tuple, TypedDict, Dict, Any

from langgraph.graph import StateGraph, END
from langchain_core.documents import Document

from logger import get_logger
from config import MAX_RETRIES

from src.retrieval.query_reformulator import QueryRewriter
from src.utils.confidence import check_confidence
from src.guardrails.prompt_injection import PromptInjectionGuard
logger = get_logger(__name__)


class GraphState(TypedDict):
    query: str
    original_query: str

    rewrite_history: List[str]

    retries: int

    retrieved_docs: List[Tuple[Document, float]]

    confidence: float
    is_confident: bool

    blocked: bool
    guardrail_response: str
    guardrail_metadata: Dict[str, Any]

    answer: str


def build_graph(retriever, llm):
    """
    Build the LangGraph workflow for adaptive RAG.
    """

    rewriter = QueryRewriter(llm)

    prompt_guard = PromptInjectionGuard()

    # -------------------------
    # Prompt Injection Guard
    # -------------------------

    def prompt_injection_node(state: GraphState) -> GraphState:
        logger.info("Running prompt injection guardrail...")

        result = prompt_guard.check(state["query"])

        if result["blocked"]:
            logger.warning("Prompt injection detected. Blocking request.")

            return {
                **state,
                "blocked": True,
                "guardrail_response": (
                    "Your request appears to contain prompt injection "
                    "or instruction manipulation."
                ),
                "guardrail_metadata": result,
            }

        return {
            **state,
            "blocked": False,
            "guardrail_metadata": result,
        }

    # -------------------------
    # Retrieve
    # -------------------------

    def retrieve_node(state: GraphState) -> GraphState:
        logger.info(f"Retrieving documents for: '{state['query']}'")

        results = retriever.retrieve(state["query"])

        return {
            **state,
            "retrieved_docs": results,
        }

    # -------------------------
    # Confidence Check
    # -------------------------

    def confidence_node(state: GraphState) -> GraphState:
        is_confident, score = check_confidence(state["retrieved_docs"])

        logger.info(
            f"Confidence Score: {score:.3f} | "
            f"Confident: {is_confident}"
        )

        return {
            **state,
            "confidence": score,
            "is_confident": is_confident,
        }

    # -------------------------
    # Query Reformulation
    # -------------------------

    def reformulate_node(state: GraphState) -> GraphState:
        logger.info("Low confidence. Reformulating query...")

        new_query = rewriter.reformulate_query(
            original_query=state["original_query"],
            previous_queries=state["rewrite_history"],
        )

        logger.info(f"Rewritten query: '{new_query}'")

        # Create a new history (don't mutate the existing one)
        rewrite_history = state["rewrite_history"]

        if new_query not in rewrite_history:
            rewrite_history = [*rewrite_history, new_query]
        else:
            logger.warning("Duplicate reformulation generated.")

        return {
            **state,
            "query": new_query,
            "rewrite_history": rewrite_history,
            "retries": state["retries"] + 1,
        }

    # -------------------------
    # Answer Generation
    # -------------------------

    def generate_node(state: GraphState) -> GraphState:
        logger.info("Generating final answer...")

        context = "\n\n".join(
            doc.page_content
            for doc, _ in state["retrieved_docs"]
        )

        prompt = f"""
            Answer the question using ONLY the provided context.

            If the answer cannot be found in the context,
            say that the information is unavailable.

            Context:
            {context}

            Question:
            {state["original_query"]}
        """

        response = llm.invoke(prompt)

        return {
            **state,
            "answer": response.content,
        }

    # -------------------------
    # Routing
    # -------------------------
    def blocked_node(state: GraphState) -> GraphState:
        logger.info("Returning guardrail response.")

        return {
            **state,
            "answer": state["guardrail_response"],
        }

    def route_after_guardrail(state: GraphState) -> str:
        if state["blocked"]:
            return "blocked"

        return "retrieve"

    def route_after_confidence(state: GraphState) -> str:
        """
        Decide whether to generate an answer or
        retry retrieval with a reformulated query.
        """

        if state["is_confident"]:
            logger.info("Confidence sufficient. Generating answer.")
            return "generate"

        if state["retries"] >= MAX_RETRIES:
            logger.info("Maximum retries reached. Generating answer.")
            return "generate"

        logger.info("Retrying with reformulated query.")
        return "reformulate"

    # -------------------------
    # Graph
    # -------------------------

    graph = StateGraph(GraphState)
    graph.add_node("prompt_injection", prompt_injection_node)
    graph.add_node("blocked", blocked_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("check_confidence", confidence_node)
    graph.add_node("reformulate", reformulate_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("prompt_injection")

    graph.add_conditional_edges(
        "prompt_injection",
        route_after_guardrail,
        {
            "retrieve": "retrieve",
            "blocked": "blocked",
        },
    )

    graph.add_edge("blocked", END)

    graph.add_edge("retrieve", "check_confidence")

    graph.add_conditional_edges(
        "check_confidence",
        route_after_confidence,
        {
            "generate": "generate",
            "reformulate": "reformulate",
        },
    )

    graph.add_edge("reformulate", "retrieve")

    graph.add_edge("generate", END)

    return graph.compile()