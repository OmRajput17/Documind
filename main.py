from src.ingestion.loader import Ingestion
from src.ingestion.chunking import Chunker


def main():

    ### Document Loading
    ingestion = Ingestion()
    documents = ingestion.load_pdf()

    print(f"\n--- Results ---")
    print(f"Total pages loaded: {len(documents)}")

    if documents:
        first = documents[0]
        print(f"\nFirst page preview:")
        print(f"  Source : {first.metadata.get('source')}")
        print(f"  Page   : {first.metadata.get('page')}")
        # print(f"  Content: {first.page_content[:200].strip()}...")

    ### Chunking the Docs
    chunker = Chunker()
    chunks = chunker.chunk_documents(documents=documents)

    print(f"\n--- Chunking Results ---")
    print(f"Total chunks : {len(chunks)}")

    if chunks:
        print(f"\nSample chunks (first 3):")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n  [Chunk {i+1}]")
            print(f"  Source  : {chunk.metadata.get('source')}")
            print(f"  Page    : {chunk.metadata.get('page')}")
            print(f"  Length  : {len(chunk.page_content)} chars")
            # print(f"  Preview : {chunk.page_content[:150].strip()}...")

    

if __name__ == "__main__":
    main()
