from src.ingestion.loader import Ingestion
from src.ingestion.chunking import Chunker
from src.ingestion.embeddings import get_embeddings_model
from src.vector_store.vectorstore import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.rrf import ReciprocalRankFusion
from src.retrieval.reranker import Reranker
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.prompt_builder import PromptBuilder
from src.generation.generator import Generator
from src.generation.formatter import ResponseFormatter
from src.pipeline.rag_pipeline import RAGPipeline
from src.test.queries import rag_pipeline_test_queries


def test_rag_pipeline():
    # ------------------------------------------------------------------ #
    # Setup                                                                 #
    # ------------------------------------------------------------------ #
    ingestion = Ingestion()
    documents = ingestion.load_pdf()

    chunker = Chunker()
    chunks = chunker.chunk_documents(documents)

    bm25 = BM25Retriever(documents=chunks)

    embedding_model = get_embeddings_model()
    vectorstore = VectorStore(embedding_model=embedding_model).load_vector_store()
    dense = DenseRetriever(vectorstore=vectorstore)

    rrf      = ReciprocalRankFusion()
    reranker = Reranker()

    retriever = HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
        rrf=rrf,
        reranker=reranker,
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        prompt_builder=PromptBuilder(),
        generator=Generator(),
        formatter=ResponseFormatter(),
    )

    # ------------------------------------------------------------------ #
    # Run queries                                                           #
    # ------------------------------------------------------------------ #
    for i, query in enumerate(rag_pipeline_test_queries, start=1):
        response = pipeline.run(query=query)

        print("=" * 80)
        print(f"Test   #{i}")
        print(f"Query  : {query}")
        print(f"\nAnswer :\n{response.answer}")
        print(f"\nSources: {len(response.sources)} cited")

        for rank, src in enumerate(response.sources, start=1):
            print(
                f"  [{rank}] {src.source}  "
                f"p.{src.page}  "
                f"score={src.relevance_score}"
            )

    print("=" * 80)
    print("RAG pipeline test complete.")


if __name__ == "__main__":
    test_rag_pipeline()
