from typing import List, Tuple
from langchain_core.documents import Document

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.rrf import ReciprocalRankFusion
from src.retrieval.reranker import Reranker


from config import (
    HYBRID_TOP_K,
    RRF_TOP_N
)


from logger import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """
    Orchestrates dense + BM25 retrieval, RRF fusion, and cross-encoder
    reranking into a single retrieval call.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        rrf: ReciprocalRankFusion,
        reranker: Reranker,
        rrf_top_n: int = RRF_TOP_N,
        hybrid_top_k: int = HYBRID_TOP_K,
    ):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf = rrf
        self.reranker = reranker
        self.rrf_top_n = rrf_top_n
        self.hybrid_top_k = hybrid_top_k

    def retrieve(self, query: str) -> List[Tuple[Document, float]]:
        """
        Retrieve the most relevant documents.

        Args:
            query: User query.

        Returns:
            List of reranked documents.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        logger.info(
            f"Running hybrid retrieval for query: '{query}'"
        )

        try:
            #--------------------------------------
            # Dense Retriever
            #--------------------------------------
            dense_results = self.dense_retriever.retrieve(
                query=query
            )

            logger.info(
                f"Dense Retriever returned {len(dense_results)} chunks."
            )


            #--------------------------------------
            # BM25 Retriever
            #--------------------------------------
            bm25_results = self.bm25_retriever.retrieve(
                query=query
            )

            logger.info(
                f"BM25 Retriever returned {len(bm25_results)} chunks."
            )

            #--------------------------------------
            # Reciprocal Rank Fusion
            #--------------------------------------
            fused_results = self.rrf.fuse(
                dense_results, 
                bm25_results
            )

            logger.info(
                f"RRF fused into {len(fused_results)} chunks."
            )

            fused_results = fused_results[:self.rrf_top_n]

            #--------------------------------------
            # Reranking Fused Results
            #--------------------------------------
            reranked_results = self.reranker.rerank(
                query=query,
                results=fused_results,
                top_n=self.hybrid_top_k
            )

            logger.info(
                f"Reranker returned {len(reranked_results)} chunks."
            )

            logger.info(
                f"Hybrid Retrieval completed successfully. "
                f"Returned {len(reranked_results)} documents."
            )

            return reranked_results

        
        except Exception:
            logger.exception(
                "Hybrid Retrieval failed."
            )
            raise