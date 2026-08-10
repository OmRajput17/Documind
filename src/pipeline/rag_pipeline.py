from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.prompt_builder import PromptBuilder
from src.generation.generator import Generator
from src.generation.formatter import ResponseFormatter

from src.datamodels.rag import RAGResponse

from logger import get_logger

logger = get_logger(__name__)

class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        prompt_builder: PromptBuilder,
        generator: Generator,
        formatter: ResponseFormatter
    ):
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.generator = generator
        self.formatter = formatter

    def run(self, query: str) -> RAGResponse:
        """
        Execute the complete RAG pipeline.

        Args:
            query: User query.

        Returns:
            Formatted response.
        """

        logger.info(
            f"Running RAG pipeline for query: {query}."
        )

        try:
            if not query.strip():
                raise ValueError("Query cannot be empty.")
            # --------------------------------------------------
            # Retrieval
            # --------------------------------------------------
            retrieved_docs = self.retriever.retrieve(query=query)
            
            logger.info(
                "Retrieved %d document chunks.",
                len(retrieved_docs)
            )
            
            # --------------------------------------------------
            # Prompt Builder
            # --------------------------------------------------
            prompt = self.prompt_builder.build(
                query=query,
                retrieved_docs=retrieved_docs
            )

            # --------------------------------------------------
            # Generation
            # --------------------------------------------------
            answer = self.generator.generate(
                prompt=prompt
            )

            # --------------------------------------------------
            # Response Formatter
            # --------------------------------------------------
            response = self.formatter.format(
                answer=answer,
                retrieved_docs=retrieved_docs
            )

            logger.info(
                "RAG pipeline completed successfully."
            )

            return response

        except Exception:
            logger.exception(
                "RAG Pipeline failed."
            )
            raise