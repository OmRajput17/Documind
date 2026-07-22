from typing import List, Tuple, TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.documents import Document

from logger import get_logger
from config import MAX_RETRIES

from src.retrieval.query_reformulator import QueryRewriter
from src.utils.confidence import check_confidence

logger = get_logger(__name__)


class GraphState(TypedDict):
    query: str
    original_query: str

    retries: int

    retrieved_docs: List[Tuple[Document, float]]

    confidence: float
    is_confident: bool

    answer: str


def build_graph(retriever, llm):
    """
    Build the LangGraph workflow for adaptive RAG.
    """

    rewriter = QueryRewriter(llm)

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

        new_query = rewriter.reformulate_query(state["query"])

        return {
            **state,
            "query": new_query,
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

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("check_confidence", confidence_node)
    graph.add_node("reformulate", reformulate_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")

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