from langchain_groq import ChatGroq

from config import (
    LLM_MODEL, 
    VALIDATION_LLM, 
    VALIDATION_TEMPERATURE, 
    GENERATION_TEMPERATURE,
    GROQ_API_KEY
)
from logger import get_logger

from functools import lru_cache

logger = get_logger(__name__)

def _load_model(model: str, temperature: float):
    try:
        logger.info("Loading LLM '%s'", model)
        return ChatGroq(
            model=model,
            temperature=temperature,
            api_key = GROQ_API_KEY
        )
    except Exception:
        logger.exception("Failed to load model '%s'.", model)
        raise

@lru_cache(maxsize=1)
def get_validation_llm():
    return _load_model(
        model=VALIDATION_LLM,   
        temperature=VALIDATION_TEMPERATURE,
    )

@lru_cache(maxsize=1)
def get_generation_llm():
    return _load_model(
        model=LLM_MODEL,
        temperature=GENERATION_TEMPERATURE,
    )

