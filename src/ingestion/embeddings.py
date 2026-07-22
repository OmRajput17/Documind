from langchain_huggingface import HuggingFaceEmbeddings
from logger import get_logger

from config import EMBEDDING_MODEL_NAME, MODEL_CACHE_PATH

logger = get_logger(__name__)

def get_embeddings_model(model_name: str = EMBEDDING_MODEL_NAME):
    logger.info(f"Loading embedding model: {model_name}")
    logger.info(f"Cache directory: {MODEL_CACHE_PATH}")

    return HuggingFaceEmbeddings(
        model_name=model_name,
        cache_folder=str(MODEL_CACHE_PATH),
    )
