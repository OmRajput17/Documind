import time
from typing import List

from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_core.documents import Document

from config import DATA_FOLDER
from logger import get_logger

logger = get_logger(__name__)


class Ingestion:
    def __init__(self, file_path: str = DATA_FOLDER):
        self.file_path = file_path

    def load_pdf(self) -> List[Document]:
        logger.info(f"Loading PDF : {self.file_path}")
        start = time.time()

        try:
            loader = DirectoryLoader(
                path=self.file_path,
                glob='**/*.pdf',
                loader_cls=PyMuPDFLoader,
                show_progress=True,
                use_multithreading=True,
                silent_errors=True
            )

            documents = loader.load()

            if not documents:
                logger.warning("No PDFs found.")
                return []
            
            unique_files = {doc.metadata.get("source") for doc in documents}

            logger.info(
                f"Loaded {len(unique_files)} PDF files "
                f"containing {len(documents)} pages "
                f"in {time.time() - start:.2f} seconds."
            )

            return documents

        except Exception as e:
            logger.exception(f"Failed to load {self.file_path} : {e}")
            raise
