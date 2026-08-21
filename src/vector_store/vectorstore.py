#src/vector_store/vectorstore.py

import time
from typing import List
from langchain_postgres import PGVector
from langchain_core.documents import Document
import psycopg

from config import PG_CONNECTION_STRING, PG_COLLECTION_NAME
from logger import get_logger


logger = get_logger(__name__)


class VectorStore:
    def __init__(self, embedding_model, connection_string: str = PG_CONNECTION_STRING,
                 collection_name: str = PG_COLLECTION_NAME):
        self.embedding_model = embedding_model
        self.connection_string = connection_string
        self.collection_name = collection_name
        self.store = None

    def build_vector_store(self, chunks: List[Document]) -> PGVector:
        """
        Create a pgvector-backed store, replacing any existing collection
        of the same name to avoid duplicates on re-runs.

        Args:
            chunks: List of chunked documents.

        Returns:
            PGVector vector store instance.
        """

        if not chunks:
            logger.warning("No chunks provided to build vector store.")
            raise ValueError("Chunks list cannot be empty.")

        logger.info(f"Building Vector Store with {len(chunks)} chunks...")
        start = time.time()

        try:
            self.store = PGVector.from_documents(
                documents=chunks,
                embedding=self.embedding_model,
                collection_name=self.collection_name,
                connection=self.connection_string,
                use_jsonb=True,
                pre_delete_collection=True,
                embedding_length=768,  # BAAI/bge-base-en-v1.5
            )

            logger.info(
                f"Vector store created successfully in "
                f"{time.time() - start:.2f}s "
                f"(collection='{self.collection_name}')."
            )

            self._create_hnsw_index()

            return self.store

        except Exception:
            logger.exception("Failed to build vector store.")
            raise

    def _create_hnsw_index(self):
        """
        Create an HNSW index on the embedding column for fast cosine-similarity search.
        PGVector stores actual vectors in the `langchain_pg_embedding` table.
        Safe to call repeatedly — uses IF NOT EXISTS.
        """
        logger.info("Creating HNSW Index (cosine distance) on embedding column...")
        start = time.time()

        try:
            conninfo = self.connection_string.replace("postgresql+psycopg://", "postgresql://")

            with psycopg.connect(conninfo=conninfo) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS langchain_pg_embedding_hnsw_index
                        ON langchain_pg_embedding
                        USING hnsw ((embedding::vector(768)) vector_cosine_ops);
                        """
                    )
                conn.commit()

                logger.info(f"HNSW index created/verified in {time.time() - start:.2f}s.")

        except Exception:
            logger.exception("Failed to create HNSW index.")
            raise

    def load_vector_store(self) -> PGVector:
        """
        Load an existing pgvector store.
        """

        logger.info(f"Connecting to vector store collection '{self.collection_name}'...")

        try:
            self.store = PGVector(
                embeddings=self.embedding_model,
                collection_name=self.collection_name,
                connection=self.connection_string,
                use_jsonb=True,
                embedding_length=768,  # BAAI/bge-base-en-v1.5
            )

            logger.info("Vector store loaded successfully.")
            return self.store

        except Exception:
            logger.exception("Failed to load vector store.")
            raise

