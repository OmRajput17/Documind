from src.ingestion.loader import Ingestion
from src.ingestion.chunking import Chunker
from src.ingestion.embeddings import get_embeddings_model
from src.vector_store.vectorstore import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.rrf import ReciprocalRankFusion
from src.retrieval.reranker import Reranker
from src.test.queries import hybrid_retriever_test_queries


def test_hybrid_retrieval_pipeline():
    # ------------------------------------------------------------------ #
    # Setup                                                                 #
    # ------------------------------------------------------------------ #
    ingestion = Ingestion()
    documents = ingestion.load_pdf()

    chunker = Chunker()
    chunks = chunker.chunk_documents(documents)

    # BM25 — built from chunks directly
    bm25 = BM25Retriever(documents=chunks)

    # Dense — loads the persisted Chroma vector store
    embedding_model = get_embeddings_model()
    vectorstore = VectorStore(embedding_model=embedding_model).load_vector_store()
    dense = DenseRetriever(vectorstore=vectorstore)

    # RRF + Reranker
    rrf     = ReciprocalRankFusion()
    reranker = Reranker()

    # ------------------------------------------------------------------ #
    # Run queries                                                           #
    # ------------------------------------------------------------------ #
    for i, query in enumerate(hybrid_retriever_test_queries, start=1):
        bm25_results  = bm25.retrieve(query)
        dense_results = dense.retrieve(query)
        fused_results = rrf.fuse(bm25_results, dense_results)
        final_results = reranker.rerank(query, fused_results)

        print("=" * 80)
        print(f"Test   #{i}")
        print(f"Query  : {query}")
        print(
            f"BM25   : {len(bm25_results)} chunks  |  "
            f"Dense  : {len(dense_results)} chunks  |  "
            f"Fused  : {len(fused_results)} chunks  |  "
            f"Reranked: {len(final_results)} chunks"
        )

        for rank, (doc, score) in enumerate(final_results, start=1):
            source  = doc.metadata.get("source", "unknown")
            page    = doc.metadata.get("page", "?")
            snippet = doc.page_content[:150].strip().replace("\n", " ")
            print(f"\n  [{rank}] Rerank Score : {score:.4f}")
            print(f"       Source       : {source}")
            print(f"       Page         : {page}")
            print(f"       Snippet      : {snippet}...")

    print("=" * 80)
    print("Hybrid retrieval pipeline test complete.")


if __name__ == "__main__":
    test_hybrid_retrieval_pipeline()
