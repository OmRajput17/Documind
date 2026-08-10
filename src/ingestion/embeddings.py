from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from logger import get_logger

from config import EMBEDDING_MODEL_NAME, MODEL_CACHE_PATH

logger = get_logger(__name__)


@lru_cache(maxsize=4)
def get_embeddings_model(model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddings:
    """
    Load and return a HuggingFaceEmbeddings model.

    Results are cached by model_name — the same model instance is reused
    across all callers within a process, avoiding repeated loads.
    """
    logger.info(f"Loading embedding model: {model_name}")
    logger.info(f"Cache directory: {MODEL_CACHE_PATH}")

    return HuggingFaceEmbeddings(
        model_name=model_name,
        cache_folder=str(MODEL_CACHE_PATH),
    )
