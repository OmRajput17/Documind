import shutil
import time
from pathlib import Path
from typing import List
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import VECTOR_STORE_PATH
from logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    def __init__(self, embedding_model, persist_directory: str = str(VECTOR_STORE_PATH)):
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.store = None

    def build_vector_store(self, chunks: List[Document]) -> Chroma:
        """
        Create and persist a Chroma vector store.
        Clears any existing store at the path before building to avoid duplicates.

        Args:
            chunks: List of chunked documents.

        Returns:
            Chroma vector store instance.
        """

        if not chunks:
            logger.warning("No chunks provided to build vector store.")
            raise ValueError("Chunks list cannot be empty.")

        # Wipe existing store to prevent duplicate entries on re-runs
        store_path = Path(self.persist_directory)
        if store_path.exists():
            logger.info(f"Clearing existing vector store at '{self.persist_directory}'...")
            shutil.rmtree(store_path)

        logger.info(f"Building Vector Store with {len(chunks)} chunks...")
        start = time.time()

        try:
            self.store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding_model,
                persist_directory=self.persist_directory,
                collection_metadata={"hnsw:space": "cosine"},
            )

            logger.info(
                f"Vector store created successfully in "
                f"{time.time() - start:.2f}s "
                f"and persisted to '{self.persist_directory}'."
            )

            return self.store

        except Exception:
            logger.exception("Failed to build vector store.")
            raise

    def load_vector_store(self) -> Chroma:
        """
        Load an existing Chroma vector store.
        """

        logger.info(f"Loading vector store from '{self.persist_directory}'...")

        try:
            self.store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model,
                collection_metadata={"hnsw:space": "cosine"},
            )

            logger.info("Vector store loaded successfully.")
            return self.store

        except Exception:
            logger.exception("Failed to load vector store.")
            raise

