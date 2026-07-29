from src.test.queries import prompt_injection_test_queries
from src.guardrails.prompt_injection import PromptInjectionGuard

prompt_injection_guard = PromptInjectionGuard()

def test_prompt_injection():
    for i, query in enumerate(prompt_injection_test_queries, start = 1):
        result = prompt_injection_guard.check(query=query)

        print("=" * 80)
        print(f"Test #{i}")
        print(f"Query : {query}")
        print(f"Action: {result.action}")
        print(f"Score: {result.score}")
        print("\nMatched Patterns:")

        if result.matched_patterns:
            for idx, pattern in enumerate(result.matched_patterns, start=1):
                print(f"\n  [{idx}]")
                print(f"  Category : {pattern.category}")
                print(f"  Pattern  : {pattern.pattern}")
                print(f"  Match    : {pattern.matched_text}")
                print(f"  Weight   : {pattern.weight}")
        else:
            print("  None")

        print(f"Reason: {result.reason}")

if __name__ == "__main__":
    test_prompt_injection()