from src.test.queries import pii_masking_test_queries
from src.guardrails.pii_masking import PIIMaskingGuard

pii_guard = PIIMaskingGuard()

def test_pii_masking():
    for i, query in enumerate(pii_masking_test_queries, start=1):
        result = pii_guard.check(query=query)

        print("=" * 80)
        print(f"Test #{i}")
        print(f"Original : {repr(query)}")
        print(f"Masked   : {result.masked_query}")
        print(f"Detected : {result.pii_detected}")
        print(f"Entities : {[e.type for e in result.entities]}")
    
if __name__ == "__main__":
    test_pii_masking()