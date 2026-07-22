from src.ingestion.loader import Ingestion
from src.ingestion.chunking import Chunker
from src.ingestion.embeddings import get_embeddings_model
from src.vector_store.vectorstore import VectorStore
from src.retrieval.retriever import Retriever
from src.orchestrator.graph import build_graph


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

    ### Chunking
    chunker = Chunker()
    chunks = chunker.chunk_documents(documents=documents)

    print(f"\n--- Chunking Results ---")
    print(f"Total chunks : {len(chunks)}")

    ### Embeddings + Vector Store
    embedding_model = get_embeddings_model()

    vs = VectorStore(embedding_model=embedding_model)
    store = vs.build_vector_store(chunks=chunks)

    ### Graph Test
    from langchain_groq import ChatGroq
    from config import GROQ_API_KEY, MODEL

    llm = ChatGroq(model=MODEL, api_key=GROQ_API_KEY)

    retriever = Retriever(vectorstore=store)
    graph = build_graph(retriever=retriever, llm=llm)

    test_query = "What is the candidate's experience with AI?"

    print(f"\n--- Graph Test ---")
    print(f"Query: {test_query}\n")

    result = graph.invoke({
        "query": test_query,
        "original_query": test_query,
        "retries": 0,
        "retrieved_docs": [],
        "confidence": 0.0,
        "is_confident": False,
        "answer": "",
    })

    print(f"  Retries       : {result['retries']}")
    print(f"  Confidence    : {result['confidence']:.3f}")
    print(f"  Final Query   : {result['query']}")
    print(f"\n  Answer:\n  {result['answer']}")


if __name__ == "__main__":
    main()
