import asyncio
import time

from langchain_chroma import Chroma

from src.ingestion.loader import Ingestion
from src.ingestion.chunking import Chunker
from src.ingestion.embeddings import get_embeddings_model

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.rrf import ReciprocalRankFusion
from src.retrieval.reranker import Reranker
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.confidence import ConfidenceEvaluator
from src.retrieval.query_reformulator import QueryRewriter

from src.generation.prompt_builder import PromptBuilder
from src.generation.generator import Generator
from src.generation.formatter import ResponseFormatter

from src.guardrails.input_validation import InputValidator
from src.guardrails.pii_masking import PIIMaskingGuard
from src.guardrails.prompt_injection import PromptInjectionGuard
from src.guardrails.nemo_guard import NemoGuard

from src.orchestrator.graph import RAGGraph

from src.test.queries import main_pipeline_test_queries, graph_smoke_test_queries

from src.utils.get_llm import get_generation_llm

from config import (
    VECTOR_STORE_PATH,
    CONFIDENCE_THRESHOLD,
    CONFIDENCE_TOP_K,
)

from logger import get_logger
from dotenv import load_dotenv

load_dotenv()

logger = get_logger(__name__)


# ------------------------------------------------------------------ #
# Helpers                                                               #
# ------------------------------------------------------------------ #

def _print_trace(timings: dict, total: float):
    width = 22
    print()
    print("=" * 50)
    print("DocuMind Pipeline Trace")
    print("=" * 50)
    print()
    for stage, elapsed in timings.items():
        print(f"  {stage:<{width}}: {elapsed:.3f} s")
    print()
    print("-" * 42)
    print(f"  {'Total':<{width}}: {total:.3f} s")
    print("=" * 50)


def _tick() -> float:
    return time.perf_counter()


# ------------------------------------------------------------------ #
# Setup                                                                 #
# ------------------------------------------------------------------ #

def build_graph() -> RAGGraph:
    print("\nInitializing DocuMind...\n")

    # ---------------------------------------------------------------- #
    # Ingestion & chunking                                               #
    # ---------------------------------------------------------------- #
    ingestion = Ingestion()
    documents = ingestion.load_pdf()

    chunker = Chunker()
    chunks = chunker.chunk_documents(documents)

    # ---------------------------------------------------------------- #
    # Vector store                                                       #
    # ---------------------------------------------------------------- #
    embedding_model = get_embeddings_model()

    vectorstore = Chroma(
        persist_directory=str(VECTOR_STORE_PATH),
        embedding_function=embedding_model,
    )

    if vectorstore._collection.count() == 0:
        logger.info(
            "Vector store empty — building and persisting at %s.",
            VECTOR_STORE_PATH,
        )
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=str(VECTOR_STORE_PATH),
        )
        logger.info(
            "Vector store built with %d chunks.",
            vectorstore._collection.count(),
        )
    else:
        logger.info(
            "Loaded existing vector store from %s (%d chunks).",
            VECTOR_STORE_PATH,
            vectorstore._collection.count(),
        )

    # ---------------------------------------------------------------- #
    # Retriever stack                                                    #
    # ---------------------------------------------------------------- #
    bm25     = BM25Retriever(documents=chunks)
    dense    = DenseRetriever(vectorstore=vectorstore)
    rrf      = ReciprocalRankFusion()
    reranker = Reranker()

    retriever = HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
        rrf=rrf,
        reranker=reranker,
    )

    # ---------------------------------------------------------------- #
    # Confidence evaluator & query rewriter                             #
    # ---------------------------------------------------------------- #
    confidence_evaluator = ConfidenceEvaluator(
        top_k=CONFIDENCE_TOP_K,
        threshold=CONFIDENCE_THRESHOLD,
    )

    query_rewriter = QueryRewriter(
        llm=get_generation_llm(),
    )

    # ---------------------------------------------------------------- #
    # Guardrails                                                         #
    # ---------------------------------------------------------------- #
    validator  = InputValidator()
    pii_masker = PIIMaskingGuard()
    regex_guard = PromptInjectionGuard()
    nemo_guard  = NemoGuard()

    # ---------------------------------------------------------------- #
    # RAGGraph                                                           #
    # ---------------------------------------------------------------- #
    graph = RAGGraph(
        validator=validator,
        pii_masker=pii_masker,
        regex_guard=regex_guard,
        nemo_guard=nemo_guard,
        retriever=retriever,
        confidence_evaluator=confidence_evaluator,
        query_rewriter=query_rewriter,
        prompt_builder=PromptBuilder(),
        generator=Generator(),
        formatter=ResponseFormatter(),
        max_reformulations=3,
    )

    return graph


# ------------------------------------------------------------------ #
# Main query loop                                                       #
# ------------------------------------------------------------------ #

async def run(graph: RAGGraph):
    print("\nDocuMind is ready.\n")

    for i, query in enumerate(graph_smoke_test_queries, start=1):
        print("\n" + "=" * 50)
        print(f"Query #{i}: {query}")
        print("=" * 50)

        t0 = _tick()

        try:
            result = await graph.invoke(
                query,
                config={
                    "tags": [
                        "documind",
                        "rag",
                        "smoke-test",
                    ],
                    "metadata": {
                        "test_type": "smoke_test",
                        "query_index": i,
                    },
                },
            )
        except Exception as e:
            logger.exception("Graph invocation failed.")
            print(f"\n[Error] {e}\n")
            continue

        total = _tick() - t0

        status = result.get("status", "UNKNOWN")

        # ---------------------------------------------------------- #
        # Blocked / error states                                       #
        # ---------------------------------------------------------- #
        if status == "BLOCKED":
            reason = (
                result.get("validation_reason")
                or result.get("nemo_reason")
                or "Query blocked."
            )
            print(f"\n[Blocked] {reason}")
            print(f"\n  Total: {total:.3f} s")
            continue

        if status == "ERROR":
            print(f"\n[Error] {result.get('error', 'Unknown error.')}")
            print(f"\n  Total: {total:.3f} s")
            continue

        # ---------------------------------------------------------- #
        # Completed — print response                                   #
        # ---------------------------------------------------------- #
        response = result.get("response")

        if response is None:
            print("\n[Warning] Pipeline completed but no response was returned.")
            continue

        print("\nAnswer")
        print("-" * 50)
        print(response.answer)

        print(f"\nSources  ({len(response.sources)} cited)")
        print("-" * 50)
        for j, src in enumerate(response.sources, start=1):
            print(
                f"  [{j}] {src.source}  "
                f"p.{src.page}  "
                f"score={src.relevance_score}"
            )

        # ---------------------------------------------------------- #
        # Trace                                                         #
        # ---------------------------------------------------------- #
        timings = {
            "Confidence":    f"{result.get('best_retrieval_confidence', 0.0):.3f}",
            "Reformulations": result.get("reformulation_count", 0),
        }

        print()
        print("=" * 50)
        print("DocuMind Pipeline Trace")
        print("=" * 50)
        print()
        print(f"  {'Confidence':<22}: {result.get('best_retrieval_confidence', 0.0):.3f}")
        print(f"  {'Reformulations':<22}: {result.get('reformulation_count', 0)}")
        print(f"  {'PII Detected':<22}: {result.get('pii_detected', False)}")
        print(f"  {'Regex Score':<22}: {result.get('regex_score', 0)}")
        print()
        print("-" * 42)
        print(f"  {'Total':<22}: {total:.3f} s")
        print("=" * 50)

    print("\n" + "=" * 50)
    print("Main pipeline test complete.")
    print("=" * 50)


def main():
    try:
        graph = build_graph()
    except Exception as e:
        print(f"\n[Fatal] Failed to initialize DocuMind: {e}")
        logger.exception("Startup failed.")
        return

    asyncio.run(run(graph))


if __name__ == "__main__":
    main()
