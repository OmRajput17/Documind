from src.ingestion.loader import Ingestion
from src.ingestion.chunking import Chunker
from src.retrieval.bm25_retriever import BM25Retriever
from src.test.queries import bm25_test_queries


def test_bm25_retriever():
    # Load and chunk documents
    ingestion = Ingestion()
    documents = ingestion.load_pdf()

    chunker = Chunker()
    chunks = chunker.chunk_documents(documents)

    # Build BM25 index
    retriever = BM25Retriever(documents=chunks)

    for i, query in enumerate(bm25_test_queries, start=1):
        results = retriever.retrieve(query)

        print("=" * 80)
        print(f"Test #{i}")
        print(f"Query  : {query}")
        print(f"Results: {len(results)} chunks retrieved")

        for rank, (doc, score) in enumerate(results, start=1):
            source = doc.metadata.get("source", "unknown")
            page   = doc.metadata.get("page", "?")
            print(f"\n  [{rank}] Score: {score:.4f}")
            print(f"       Source : {source}")
            print(f"       Page   : {page}")
            print(f"       Snippet: {doc.page_content[:50].strip()}...")


if __name__ == "__main__":
    test_bm25_retriever()
