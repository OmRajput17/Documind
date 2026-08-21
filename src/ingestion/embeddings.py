#src/ingestion/embeddings.py

from functools import lru_cache

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from logger import get_logger

from config import EMBEDDING_MODEL_NAME

logger = get_logger(__name__)


@lru_cache(maxsize=4)
def get_embeddings_model(model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceEndpointEmbeddings:
    """
    Load and return a HuggingFace Inference API embeddings client.

    Runs on HuggingFace's servers — no local model weights, no torch,
    no cache_folder needed. Requires HUGGINGFACEHUB_API_TOKEN to be set
    as an environment variable.
    """
    logger.info(f"Connecting to HuggingFace-hosted embedding model: {model_name}")

    return HuggingFaceEndpointEmbeddings(
        model=model_name,
    )
