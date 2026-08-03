import asyncio

from src.guardrails.nemo_guard import NemoGuard
from src.test.queries import nemo_guard_test_queries

async def test_nemo_guard():

    nemo_guard = NemoGuard()

    for i, query in enumerate(nemo_guard_test_queries, start=1):

        result = await nemo_guard.check(query=query)

        print("="*80)
        print(f"Test #{i}")
        print(f"Query : {query}")
        print(f"Action : {result.action}")
        print(f"Reason : {result.reason}")

if __name__=="__main__":
    asyncio.run(test_nemo_guard())