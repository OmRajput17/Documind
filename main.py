from langchain_chroma import Chroma

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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
from src.datamodels.rag import RAGResponse

from src.utils.get_llm import get_generation_llm

from src.api.schemas import QueryRequest

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

graph = build_graph()


app = FastAPI(
    title="DocuMind API",
    description="Document RAG API",
    version="1.0.0",
)

@app.get("/health")
async def health():
    return {
        "status":"healthy"
    }


@app.post("/query", response_model=RAGResponse)
async def query(request: QueryRequest):
    try:
        result = await graph.invoke(
            request.query,
            config={
                "tags":[
                    "documind",
                    "rag",
                    "api"
                ],
                "metadata":{
                    "test_type":"api"
                },
            },
        )

    except Exception as e:
        logger.exception("Graph Invocation failed.")
        raise HTTPException(
            status_code=500,
            detail="RAG Pipeline Failed."
        )

    status = result.get("status", "UNKNOWN")

    if status == "BLOCKED":
        return {
            "status":"BLOCKED",
            "reason":(
                result.get("validation_reason")
                or result.get("nemo_reason")
                or "Query Blocked"
            )
        }
    
    if status == "ERROR":
        raise HTTPException(
            status_code=500,
            detail=result.get(
                "error",
                "Unknown Pipeline Error."
            )
        )

    response = result.get("response")

    if response is None:
        raise HTTPException(
            status_code=500,
            detail="Pipeline Completed without a response."
        )

    return response