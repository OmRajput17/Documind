import time
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import CHUNK_SIZE, CHUNK_OVERLAP, TEXT_SEPARATORS
from logger import get_logger

logger = get_logger(__name__)

class Chunker:
    def __init__(self, chunk_size : int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=TEXT_SEPARATORS,
            add_start_index=True
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into overlapping chunks while preserving metadata.

        Args:
            documents: List of LangChain Document objects.

        Returns:
            List[Document]: Chunked documents with metadata.
        """

        if not documents:
            logger.warning("No documents received for chunking.")
            return []

        logger.info(
            f"Chunking {len(documents)} documents "
            f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})..."
        )

        start = time.time()

        try:
            chunks = self.splitter.split_documents(documents)

            for idx, chunk in enumerate(chunks):
                source = chunk.metadata.get("source", "unknown")
                page = chunk.metadata.get("page", 0)

                chunk.metadata["chunk_id"] = f"{source}_page{page}_chunk{idx}"
                chunk.metadata["chunk_size"] = len(chunk.page_content)

            avg_chunk_size = (
                sum(chunk.metadata["chunk_size"] for chunk in chunks) / len(chunks)
            )

            logger.info(
                f"Created {len(chunks)} chunks "
                f"(avg size: {avg_chunk_size:.0f} chars) "
                f"in {time.time() - start:.2f}s."
            )

            return chunks

        except Exception:
            logger.exception("Failed to chunk documents.")
            raise