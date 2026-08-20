from src.test.queries import input_validation_test_queries
from src.guardrails.input_validation import InputValidator

def test_valid():
    for query in input_validation_test_queries:
        valid, reason = InputValidator().is_valid_query(query=query)

        print(f"Query: {query}")
        print(f"Valid: {valid}")
        print(f"Reason: {reason}")
        print("-" * 50)
    # query = "What are the RBI guidelines for NBFC registration?"
    # valid, reason = InputValidator().is_valid_query(query=query)

    # print(f"Query: {query}")
    # print(f"Valid: {valid}")
    # print(f"Reason: {reason}")
    # print("-" * 50)

if __name__ == "__main__":
    test_valid()

