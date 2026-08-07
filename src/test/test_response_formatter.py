from langchain_core.documents import Document

from src.generation.formatter import ResponseFormatter


def test_response_formatter():
    formatter = ResponseFormatter()

    test_cases = [
        # ------------------------------------------------------------------ #
        # Case 1 — single source with full metadata
        # ------------------------------------------------------------------ #
        {
            "label": "Single source with full metadata",
            "answer": "The Transformer uses self-attention to relate positions in a sequence.",
            "docs": [
                (
                    Document(
                        page_content="The Transformer model relies on self-attention mechanisms.",
                        metadata={"source": "illustrated_transformer.pdf", "page": 3},
                    ),
                    0.9231,
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # Case 2 — multiple sources
        # ------------------------------------------------------------------ #
        {
            "label": "Multiple sources",
            "answer": "Om Rajput has built RAG pipelines and AI agents using LangChain and ChromaDB.",
            "docs": [
                (
                    Document(
                        page_content="Built a multi-document RAG pipeline with LangChain.",
                        metadata={"source": "Om_Rajput_AI_Developer.pdf", "page": 1},
                    ),
                    0.8754,
                ),
                (
                    Document(
                        page_content="Used ChromaDB as the vector store for semantic retrieval.",
                        metadata={"source": "Om_Rajput_AI_Developer.pdf", "page": 2},
                    ),
                    0.7612,
                ),
                (
                    Document(
                        page_content="Designed agentic workflows with LangGraph.",
                        metadata={"source": "Om_Rajput_SDE.pdf", "page": 1},
                    ),
                    0.6891,
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # Case 3 — missing metadata (fallback values)
        # ------------------------------------------------------------------ #
        {
            "label": "Missing metadata (fallback to Unknown / N/A)",
            "answer": "Positional encoding injects order information into the embeddings.",
            "docs": [
                (
                    Document(
                        page_content="Positional encodings are added to token embeddings.",
                        metadata={},
                    ),
                    0.5012,
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # Case 4 — empty retrieved docs
        # ------------------------------------------------------------------ #
        {
            "label": "No retrieved documents",
            "answer": "I couldn't find that information in the uploaded documents.",
            "docs": [],
        },

        # ------------------------------------------------------------------ #
        # Case 5 — score rounding
        # ------------------------------------------------------------------ #
        {
            "label": "Score rounding (4+ decimal places)",
            "answer": "Multi-head attention projects queries, keys, and values h times.",
            "docs": [
                (
                    Document(
                        page_content="Multi-head attention runs the attention function in parallel.",
                        metadata={"source": "illustrated_transformer.pdf", "page": 7},
                    ),
                    0.123456789,
                ),
            ],
        },
    ]

    for i, case in enumerate(test_cases, start=1):
        result = formatter.format(
            answer=case["answer"],
            retrieved_docs=case["docs"],
        )

        print("=" * 80)
        print(f"Test #{i} — {case['label']}")
        print(f"Answer  : {result['answer']}")
        print(f"Sources : {len(result['sources'])} returned")

        for rank, src in enumerate(result["sources"], start=1):
            print(f"\n  [{rank}] Source         : {src['source']}")
            print(f"       Page           : {src['page']}")
            print(f"       Relevance Score: {src['relevance_score']}")

    print("=" * 80)
    print("ResponseFormatter test complete.")


if __name__ == "__main__":
    test_response_formatter()
