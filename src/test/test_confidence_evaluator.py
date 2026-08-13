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

from src.test.queries import confidence_test_queries

from config import VECTOR_STORE_PATH, CONFIDENCE_THRESHOLD, CONFIDENCE_TOP_K


# ------------------------------------------------------------------ #
# Retriever factory                                                     #
# ------------------------------------------------------------------ #

def build_retriever() -> HybridRetriever:

    # --------------------------------------------------
    # Load corpus
    # --------------------------------------------------
    ingestion = Ingestion()
    documents = ingestion.load_pdf()

    chunker = Chunker()
    chunks  = chunker.chunk_documents(documents)

    # --------------------------------------------------
    # Embeddings
    # --------------------------------------------------
    embedding_model = get_embeddings_model()

    # --------------------------------------------------
    # Vector store
    # --------------------------------------------------
    vectorstore = Chroma(
        persist_directory=str(VECTOR_STORE_PATH),
        embedding_function=embedding_model,
    )

    # --------------------------------------------------
    # Retrieval stack
    # --------------------------------------------------
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

    return retriever


# ------------------------------------------------------------------ #
# Test                                                                  #
# ------------------------------------------------------------------ #

def test_confidence_evaluator():

    retriever = build_retriever()

    evaluator = ConfidenceEvaluator(
        top_k=CONFIDENCE_TOP_K,
        threshold=CONFIDENCE_THRESHOLD,
    )

    print("\n")
    print("=" * 80)
    print("DocuMind — Confidence Calibration")
    print("=" * 80)

    for index, query in enumerate(confidence_test_queries, start=1):

        print("\n")
        print("=" * 80)
        print(f"QUERY #{index}")
        print("=" * 80)
        print(f"\nQuery:\n{query}")

        # --------------------------------------------------
        # Retrieval
        # --------------------------------------------------
        results = retriever.retrieve(query=query)

        if not results:
            print("\nNo results returned.")
            continue

        scores = [float(score) for _, score in results]

        # --------------------------------------------------
        # Raw retrieval signals
        # --------------------------------------------------
        top_score = scores[0]

        top_3      = scores[:3]
        top_3_mean = sum(top_3) / len(top_3)

        score_gap = (
            scores[0] - scores[1]
            if len(scores) > 1
            else scores[0]
        )

        print("\nRetrieval Signals")
        print("-" * 40)
        print(f"Top score      : {top_score:.3f}")
        print(f"Top-3 mean     : {top_3_mean:.3f}")
        print(f"Score gap      : {score_gap:.3f}")
        print(f"All scores     : {[round(s, 3) for s in scores]}")

        # --------------------------------------------------
        # ConfidenceEvaluator metrics
        # --------------------------------------------------
        metrics   = evaluator.evaluate(results)
        confident = metrics["confidence"] >= evaluator.threshold
        verdict   = "CONFIDENT" if confident else "LOW CONFIDENCE"

        print("\nConfidence Evaluator")
        print("-" * 40)
        print(f"Confidence   : {metrics['confidence']:.3f}  [{verdict}]  (threshold={evaluator.threshold})")
        print(f"Mean Top-{evaluator.top_k}  : {metrics['mean_top_k']:.3f}")
        print(f"Top-2 Mean   : {metrics['top_2_mean']:.3f}")
        print(f"Top-3 Mean   : {metrics['top_3_mean']:.3f}")
        print(f"Median       : {metrics['median_score']:.3f}")
        print(f"Std Dev      : {metrics['score_std']:.3f}")
        print(f"Score Gap    : {metrics['score_gap']:.3f}")

        # --------------------------------------------------
        # Top retrieved documents
        # --------------------------------------------------
        print("\nTop Retrieved Documents")
        print("-" * 40)

        for rank, (doc, score) in enumerate(results[:5], start=1):
            source  = doc.metadata.get("source", "Unknown")
            page    = doc.metadata.get("page", "N/A")
            snippet = doc.page_content[:700].replace("\n", " ")

            print(f"\n[{rank}] score={score:.3f}  source={source}  page={page}")
            print(snippet)

        # --------------------------------------------------
        # Manual label
        # --------------------------------------------------
        print("\n")
        relevance = input(
            "Is the retrieved context sufficient to answer this query? [y/n]: "
        ).strip().lower()

        print(
            f"MANUAL LABEL: {'RELEVANT' if relevance == 'y' else 'NOT RELEVANT'}"
        )

    print("\n")
    print("=" * 80)
    print("ConfidenceEvaluator calibration complete.")
    print("=" * 80)


if __name__ == "__main__":
    test_confidence_evaluator()
