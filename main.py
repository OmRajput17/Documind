from src.ingestion.loader import Ingestion
from src.ingestion.chunking import Chunker
from src.retrieval.embeddings import get_embeddings_model
from src.retrieval.vectorstore import VectorStore


def main():

    ### Document Loading
    ingestion = Ingestion()
    documents = ingestion.load_pdf()

    print(f"\n--- Results ---")
    print(f"Total pages loaded: {len(documents)}")

    if documents:
        first = documents[0]
        print(f"\nFirst page preview:")
        print(f"  Source : {first.metadata.get('source')}")
        print(f"  Page   : {first.metadata.get('page')}")
        # print(f"  Content: {first.page_content[:200].strip()}...")

    ### Chunking the Docs
    chunker = Chunker()
    chunks = chunker.chunk_documents(documents=documents)

    print(f"\n--- Chunking Results ---")
    print(f"Total chunks : {len(chunks)}")

    if chunks:
        print(f"\nSample chunks (first 3):")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n  [Chunk {i+1}]")
            print(f"  Source  : {chunk.metadata.get('source')}")
            print(f"  Page    : {chunk.metadata.get('page')}")
            print(f"  Length  : {len(chunk.page_content)} chars")
            # print(f"  Preview : {chunk.page_content[:150].strip()}...")

    ### Embeddings + Vector Store
    embedding_model = get_embeddings_model()

    vs = VectorStore(embedding_model=embedding_model)
    store = vs.build_vector_store(chunks=chunks)

    ### Retrieval Test
    test_query = "What is the candidate's experience with AI?"
    print(f"\n--- Retrieval Test ---")
    print(f"Query: {test_query}\n")

    results = store.similarity_search(test_query, k=3)

    for i, doc in enumerate(results):
        print(f"  [Result {i+1}]")
        print(f"  Source  : {doc.metadata.get('source')}")
        print(f"  Page    : {doc.metadata.get('page')}")
        # print(f"  Preview : {doc.page_content[:200].strip()}")
        # print()


if __name__ == "__main__":
    main()
