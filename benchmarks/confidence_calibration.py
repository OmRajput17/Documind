import csv
from pathlib import Path
from statistics import mean, median, stdev

from langchain_chroma import Chroma

from src.ingestion.loader import Ingestion
from src.ingestion.chunking import Chunker
from src.ingestion.embeddings import get_embeddings_model

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.rrf import ReciprocalRankFusion
from src.retrieval.reranker import Reranker
from src.retrieval.hybrid_retriever import HybridRetriever

from src.test.queries import confidence_test_queries

from config import VECTOR_STORE_PATH

# --------------------------------------------------
# Output
# --------------------------------------------------

OUTPUT_PATH = Path("benchmarks/results/confidence_calibration.csv")


# --------------------------------------------------
# Build retriever
# --------------------------------------------------

def build_retriever() -> HybridRetriever:
    ingestion = Ingestion()
    documents = ingestion.load_pdf()

    chunker = Chunker()
    chunks  = chunker.chunk_documents(documents)

    embedding_model = get_embeddings_model()

    vectorstore = Chroma(
        persist_directory=str(VECTOR_STORE_PATH),
        embedding_function=embedding_model,
    )

    bm25     = BM25Retriever(documents=chunks)
    dense    = DenseRetriever(vectorstore=vectorstore)
    rrf      = ReciprocalRankFusion()
    reranker = Reranker()

    return HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
        rrf=rrf,
        reranker=reranker,
    )


# --------------------------------------------------
# Calculate signals
# --------------------------------------------------

def calculate_signals(scores: list) -> dict:
    top_score = scores[0]

    top_2_mean = mean(scores[:min(2, len(scores))])
    top_3_mean = mean(scores[:min(3, len(scores))])

    k          = min(5, len(scores))
    mean_top_k = mean(scores[:k])

    median_score = median(scores)

    score_std = stdev(scores) if len(scores) > 1 else 0.0

    score_gap = (
        scores[0] - scores[1]
        if len(scores) > 1
        else scores[0]
    )

    return {
        "top_score":    top_score,
        "top_2_mean":   top_2_mean,
        "top_3_mean":   top_3_mean,
        "mean_top_k":   mean_top_k,
        "median_score": median_score,
        "score_std":    score_std,
        "score_gap":    score_gap,
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    retriever = build_retriever()

    fieldnames = [
        "query",
        "top_score",
        "top_2_mean",
        "top_3_mean",
        "mean_top_k",
        "median_score",
        "score_std",
        "score_gap",
        "relevance",
    ]

    # --------------------------------------------------
    # Resume existing calibration
    # --------------------------------------------------
    existing_queries: set[str] = set()

    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_queries.add(row["query"])

        print(f"Found {len(existing_queries)} previously evaluated queries.")

    # --------------------------------------------------
    # Open CSV in append mode
    # --------------------------------------------------
    file_exists = OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size > 0

    with OUTPUT_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        # --------------------------------------------------
        # Run queries
        # --------------------------------------------------
        for index, query in enumerate(confidence_test_queries, start=1):

            if query in existing_queries:
                print(f"\nSkipping Query #{index} (already evaluated)")
                continue

            print("\n")
            print("=" * 80)
            print(f"QUERY #{index}")
            print("=" * 80)
            print(f"\n{query}")

            # --------------------------------------------------
            # Retrieval
            # --------------------------------------------------
            results = retriever.retrieve(query=query)

            if not results:
                print("\nNo retrieval results.")
                writer.writerow({
                    "query":        query,
                    "top_score":    0.0,
                    "top_2_mean":   0.0,
                    "top_3_mean":   0.0,
                    "mean_top_k":   0.0,
                    "median_score": 0.0,
                    "score_std":    0.0,
                    "score_gap":    0.0,
                    "relevance":    "",
                })
                f.flush()
                continue

            scores  = [float(score) for _, score in results]
            signals = calculate_signals(scores)

            # --------------------------------------------------
            # Display signals
            # --------------------------------------------------
            print("\nRetrieval Signals")
            print("-" * 40)
            for key, value in signals.items():
                print(f"{key:<15}: {value:.3f}")

            # --------------------------------------------------
            # Display retrieved documents
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
            while True:
                relevance = input(
                    "\nIs the retrieved context sufficient "
                    "to answer this query? [y/n]: "
                ).strip().lower()

                if relevance in {"y", "n"}:
                    break

                print("Please enter 'y' or 'n'.")

            # --------------------------------------------------
            # Save result
            # --------------------------------------------------
            writer.writerow({
                "query": query,
                **{key: round(value, 6) for key, value in signals.items()},
                "relevance": "relevant" if relevance == "y" else "not_relevant",
            })

            f.flush()

            print(f"\nSaved to: {OUTPUT_PATH}")

    print("\n")
    print("=" * 80)
    print("Confidence calibration complete.")
    print(f"Results: {OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
