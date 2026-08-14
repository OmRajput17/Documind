from typing import List, Tuple, Optional
from typing_extensions import TypedDict

from langchain_core.documents import Document

from src.datamodels.rag import RAGResponse


class RAGState(TypedDict, total = False):
    ## Input
    query: str
    masked_query: str

    ## Input Validation
    is_valid: bool
    valition_reason: str

    ## PII
    pii_detected: bool
    pii_entities: List[str]

    ## Prompt Injections
    regex_blocked: bool
    regex_score: float

    ## NeMo Guardrails
    nemo_blocked: bool
    nemo_reason: str

    ## Retrieval
    retrieved_docs: List[Tuple[Document, float]]
    retrieval_confidence: float

    ## Best Retrieval
    best_retrieved_docs: List[Tuple[Document, float]]
    best_retrieval_confidence: float
    best_query: str

    ## Query Reformulation
    reformulated_query: str
    reformulation_count: int
    max_reformulations: int
    rewrite_history: List[str]

    ## Generation 
    prompt: str
    answer: str

    ## Final Response
    response: Optional[RAGResponse]

    ## Execution
    status: str
    error: Optional[str]