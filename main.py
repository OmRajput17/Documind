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

from src.generation.prompt_builder import PromptBuilder
from src.generation.generator import Generator
from src.generation.formatter import ResponseFormatter

from src.pipeline.rag_pipeline import RAGPipeline

from src.guardrails.input_validation import InputValidator
from src.guardrails.pii_masking import PIIMaskingGuard
from src.guardrails.prompt_injection import PromptInjectionGuard
from src.guardrails.nemo_guard import NemoGuard

from src.test.queries import main_pipeline_test_queries

from config import VECTOR_STORE_PATH

from logger import get_logger

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

def build_pipeline():
    print("\nInitializing DocuMind...\n")

    # Ingestion
    ingestion = Ingestion()
    documents = ingestion.load_pdf()

    # Chunking
    chunker = Chunker()
    chunks = chunker.chunk_documents(documents)

    # Embedding model
    embedding_model = get_embeddings_model()

    # Vector store — load if it exists and has documents, otherwise build and persist
    vectorstore = Chroma(
        persist_directory=str(VECTOR_STORE_PATH),
        embedding_function=embedding_model,
    )

    if vectorstore._collection.count() == 0:
        logger.info("Vector store is empty. Building and persisting at %s", VECTOR_STORE_PATH)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=str(VECTOR_STORE_PATH),
        )
        logger.info("Vector store built with %d chunks.", vectorstore._collection.count())
    else:
        logger.info(
            "Loaded existing vector store from %s (%d chunks).",
            VECTOR_STORE_PATH,
            vectorstore._collection.count(),
        )

    # Retriever stack
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

    # RAG pipeline
    pipeline = RAGPipeline(
        retriever=retriever,
        prompt_builder=PromptBuilder(),
        generator=Generator(),
        formatter=ResponseFormatter(),
    )

    # Guardrails
    validator       = InputValidator()
    pii_masker      = PIIMaskingGuard()
    regex_guard     = PromptInjectionGuard()
    nemo_guard      = NemoGuard()

    return pipeline, validator, pii_masker, regex_guard, nemo_guard


# ------------------------------------------------------------------ #
# Main query loop                                                       #
# ------------------------------------------------------------------ #

async def run(pipeline, validator, pii_masker, regex_guard, nemo_guard):
    print("\nDocuMind is ready.\n")

    for i, query in enumerate(main_pipeline_test_queries, start=1):
        print("\n" + "=" * 50)
        print(f"Query #{i}: {query}")
        print("=" * 50)

        timings: dict[str, float] = {}
        pipeline_start = _tick()

        # ---------------------------------------------------------- #
        # 1. Input Validation                                          #
        # ---------------------------------------------------------- #
        t0 = _tick()
        try:
            is_valid, reason = validator.is_valid_query(query)
        except Exception as e:
            print(f"\n[Error] Input validation failed: {e}\n")
            continue
        timings["Input Validation"] = _tick() - t0

        if not is_valid:
            print(f"\n[Blocked] Invalid input — {reason}\n")
            _print_trace(timings, _tick() - pipeline_start)
            continue

        # ---------------------------------------------------------- #
        # 2. PII Masking                                               #
        # ---------------------------------------------------------- #
        t0 = _tick()
        pii_result = pii_masker.check(query)
        timings["PII Masking"] = _tick() - t0

        masked_query = pii_result.masked_query

        if pii_result.pii_detected:
            detected = [e.type for e in pii_result.entities]
            print(f"\n[PII Masked] Detected: {detected}")

        # ---------------------------------------------------------- #
        # 3. Regex Guard (Prompt Injection)                            #
        # ---------------------------------------------------------- #
        t0 = _tick()
        regex_result = regex_guard.check(masked_query)
        timings["Regex Guard"] = _tick() - t0

        if regex_result.action == "BLOCK":
            print(f"\n[Blocked] Prompt injection detected (score={regex_result.score}).\n")
            _print_trace(timings, _tick() - pipeline_start)
            continue

        # ---------------------------------------------------------- #
        # 4. NeMo Guardrails                                           #
        # ---------------------------------------------------------- #
        t0 = _tick()
        try:
            nemo_result = await nemo_guard.check(masked_query)
        except Exception as e:
            logger.warning("NeMo guard failed, failing open: %s", e)
            nemo_result = None
        timings["NeMo Guardrails"] = _tick() - t0

        if nemo_result and nemo_result.action == "BLOCK":
            print(f"\n[Blocked] NeMo Guardrails — {nemo_result.reason}\n")
            _print_trace(timings, _tick() - pipeline_start)
            continue

        # ---------------------------------------------------------- #
        # 5 – 8. RAG Pipeline (with per-stage timing)                 #
        # ---------------------------------------------------------- #
        try:
            # Retrieval
            t0 = _tick()
            retrieved_docs = pipeline.retriever.retrieve(query=masked_query)
            timings["Hybrid Retrieval"] = _tick() - t0

            # Prompt building
            t0 = _tick()
            prompt = pipeline.prompt_builder.build(
                query=masked_query,
                retrieved_docs=retrieved_docs,
            )
            timings["Prompt Builder"] = _tick() - t0

            # Generation
            t0 = _tick()
            answer = pipeline.generator.generate(prompt=prompt)
            timings["Generation"] = _tick() - t0

            # Formatting
            t0 = _tick()
            response = pipeline.formatter.format(
                answer=answer,
                retrieved_docs=retrieved_docs,
            )
            timings["Response Formatter"] = _tick() - t0

        except Exception as e:
            print(f"\n[Error] Pipeline failed: {e}\n")
            logger.exception("Pipeline execution failed.")
            continue

        total = _tick() - pipeline_start

        # ---------------------------------------------------------- #
        # Output                                                        #
        # ---------------------------------------------------------- #
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

        _print_trace(timings, total)

    print("\n" + "=" * 50)
    print("Main pipeline test complete.")
    print("=" * 50)


def main():
    try:
        pipeline, validator, pii_masker, regex_guard, nemo_guard = build_pipeline()
    except Exception as e:
        print(f"\n[Fatal] Failed to initialize DocuMind: {e}")
        logger.exception("Startup failed.")
        return

    asyncio.run(run(pipeline, validator, pii_masker, regex_guard, nemo_guard))


if __name__ == "__main__":
    main()
