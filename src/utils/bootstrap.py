from src.ingestion.embeddings import get_embeddings_model
from src.vector_store.vectorstore import VectorStore
from src.retrieval.retriever import Retriever


def create_retriever() -> Retriever:
    """
    Create a fully initialized Retriever using the
    persisted Chroma vector store.
    """

    embedding_model = get_embeddings_model()

    vectorstore = VectorStore(
        embedding_model=embedding_model
    ).load_vector_store()

    return Retriever(vectorstore)