from src.ingestion.loader import Ingestion


def main():
    ingestion = Ingestion()
    documents = ingestion.load_pdf()

    print(f"\n--- Results ---")
    print(f"Total pages loaded: {len(documents)}")

    if documents:
        first = documents[0]
        print(f"\nFirst page preview:")
        print(f"  Source : {first.metadata.get('source')}")
        print(f"  Page   : {first.metadata.get('page')}")
        print(f"  Content: {first.page_content[:200].strip()}...")


if __name__ == "__main__":
    main()
