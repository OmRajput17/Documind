from typing import List

from pydantic import BaseModel


class Source(BaseModel):
    source: str
    page: int | None = None
    relevance_score: float


class RAGResponse(BaseModel):
    answer: str
    sources: List[Source]