"""
DocuMind performance benchmark.

Measures:

1. Pipeline startup
2. End-to-end query latency
3. Guardrail latency
4. Dense retrieval latency
5. BM25 retrieval latency
6. RRF latency
7. Reranker latency
8. Prompt building latency
9. Generation latency
10. Response formatting latency

Run:

    uv run python -m benchmarks.benchmark
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, List


# ------------------------------------------------------------------
# Project root
# ------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ------------------------------------------------------------------
# Project imports
# ------------------------------------------------------------------

from benchmarks.queries import (
    BENCHMARK_QUERIES,
    BENCHMARK_RUNS,
    WARMUP_RUNS,
)

from benchmarks.metrices import calculate_metrics

from logger import get_logger

from main import build_pipeline


logger = get_logger(__name__)


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

RESULTS_DIR = (
    ROOT_DIR
    / "benchmarks"
    / "results"
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def tick() -> float:
    """
    High-resolution timer.
    """
    return time.perf_counter()


def save_json(
    filename: str,
    data: dict,
) -> None:
    """
    Save benchmark result as JSON.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULTS_DIR
        / filename
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )

    logger.info(
        "Benchmark result saved to %s",
        output_path,
    )


# ------------------------------------------------------------------
# Startup benchmark
# ------------------------------------------------------------------

def benchmark_startup() -> dict:
    """
    Measure complete DocuMind initialization.

    This includes everything executed inside build_pipeline().
    """

    print()
    print("=" * 70)
    print("STARTUP BENCHMARK")
    print("=" * 70)

    start = tick()

    (
        pipeline,
        validator,
        pii_masker,
        regex_guard,
        nemo_guard,
    ) = build_pipeline()

    elapsed = tick() - start

    print()
    print(
        f"Pipeline startup: {elapsed:.4f} seconds"
    )

    return {
        "pipeline": pipeline,
        "validator": validator,
        "pii_masker": pii_masker,
        "regex_guard": regex_guard,
        "nemo_guard": nemo_guard,
        "startup_time": elapsed,
    }


# ------------------------------------------------------------------
# Guarded query execution
# ------------------------------------------------------------------

async def run_query(
    pipeline,
    validator,
    pii_masker,
    regex_guard,
    nemo_guard,
    query: str,
) -> Dict:

    timings = {}

    total_start = tick()

    # --------------------------------------------------------------
    # 1. Input validation
    # --------------------------------------------------------------

    start = tick()

    is_valid, reason = (
        validator.is_valid_query(query)
    )

    timings["input_validation"] = (
        tick() - start
    )

    if not is_valid:

        timings["total"] = (
            tick() - total_start
        )

        return {
            "status": "BLOCKED",
            "reason": reason,
            "timings": timings,
        }

    # --------------------------------------------------------------
    # 2. PII masking
    # --------------------------------------------------------------

    start = tick()

    pii_result = (
        pii_masker.check(query)
    )

    timings["pii_masking"] = (
        tick() - start
    )

    masked_query = (
        pii_result.masked_query
    )

    # --------------------------------------------------------------
    # 3. Regex prompt injection
    # --------------------------------------------------------------

    start = tick()

    regex_result = (
        regex_guard.check(masked_query)
    )

    timings["regex_guard"] = (
        tick() - start
    )

    if regex_result.action == "BLOCK":

        timings["total"] = (
            tick() - total_start
        )

        return {
            "status": "BLOCKED",
            "reason": "Regex prompt injection guard",
            "timings": timings,
        }

    # --------------------------------------------------------------
    # 4. NeMo Guardrails
    # --------------------------------------------------------------

    start = tick()

    nemo_result = await nemo_guard.check(
        masked_query
    )

    timings["nemo_guardrails"] = (
        tick() - start
    )

    if (
        nemo_result
        and nemo_result.action == "BLOCK"
    ):

        timings["total"] = (
            tick() - total_start
        )

        return {
            "status": "BLOCKED",
            "reason": nemo_result.reason,
            "timings": timings,
        }

    # --------------------------------------------------------------
    # 5. Dense retrieval
    # --------------------------------------------------------------

    start = tick()

    dense_results = (
        pipeline.retriever
        .dense_retriever
        .retrieve(
            query=masked_query
        )
    )

    timings["dense_retrieval"] = (
        tick() - start
    )

    # --------------------------------------------------------------
    # 6. BM25 retrieval
    # --------------------------------------------------------------

    start = tick()

    bm25_results = (
        pipeline.retriever
        .bm25_retriever
        .retrieve(
            query=masked_query
        )
    )

    timings["bm25_retrieval"] = (
        tick() - start
    )

    # --------------------------------------------------------------
    # 7. Reciprocal Rank Fusion
    # --------------------------------------------------------------

    start = tick()

    fused_results = (
        pipeline.retriever
        .rrf
        .fuse(
            dense_results,
            bm25_results,
        )
    )

    fused_results = fused_results[
        :pipeline.retriever.rrf_top_n
    ]

    timings["rrf"] = (
        tick() - start
    )

    # --------------------------------------------------------------
    # 8. Cross Encoder reranking
    # --------------------------------------------------------------

    start = tick()

    reranked_results = (
        pipeline.retriever
        .reranker
        .rerank(
            query=masked_query,
            results=fused_results,
            top_n=(
                pipeline.retriever
                .hybrid_top_k
            ),
        )
    )

    timings["reranking"] = (
        tick() - start
    )

    # --------------------------------------------------------------
    # 9. Prompt building
    # --------------------------------------------------------------

    start = tick()

    prompt = (
        pipeline.prompt_builder.build(
            query=masked_query,
            retrieved_docs=reranked_results,
        )
    )

    timings["prompt_builder"] = (
        tick() - start
    )

    # --------------------------------------------------------------
    # 10. LLM generation
    # --------------------------------------------------------------

    start = tick()

    answer = (
        pipeline.generator.generate(
            prompt=prompt
        )
    )

    timings["generation"] = (
        tick() - start
    )

    # --------------------------------------------------------------
    # 11. Response formatting
    # --------------------------------------------------------------

    start = tick()

    pipeline.formatter.format(
        answer=answer,
        retrieved_docs=reranked_results,
    )

    timings["response_formatter"] = (
        tick() - start
    )

    # --------------------------------------------------------------
    # Total
    # --------------------------------------------------------------

    timings["total"] = (
        tick() - total_start
    )

    return {
        "status": "SUCCESS",
        "timings": timings,
    }


# ------------------------------------------------------------------
# Warm-up
# ------------------------------------------------------------------

async def warmup(
    pipeline,
    validator,
    pii_masker,
    regex_guard,
    nemo_guard,
) -> None:

    if WARMUP_RUNS <= 0:
        return

    print()
    print(
        f"Running {WARMUP_RUNS} warm-up iteration(s)..."
    )

    for query in BENCHMARK_QUERIES[
        :WARMUP_RUNS
    ]:

        try:

            await run_query(
                pipeline,
                validator,
                pii_masker,
                regex_guard,
                nemo_guard,
                query,
            )

        except Exception:

            logger.exception(
                "Warm-up failed for query: %s",
                query,
            )


# ------------------------------------------------------------------
# Query benchmark
# ------------------------------------------------------------------

async def benchmark_queries(
    pipeline,
    validator,
    pii_masker,
    regex_guard,
    nemo_guard,
) -> dict:

    print()
    print("=" * 70)
    print("QUERY BENCHMARK")
    print("=" * 70)

    await warmup(
        pipeline,
        validator,
        pii_masker,
        regex_guard,
        nemo_guard,
    )

    query_results = []

    for query_index, query in enumerate(
        BENCHMARK_QUERIES,
        start=1,
    ):

        print()
        print(
            f"[{query_index}/{len(BENCHMARK_QUERIES)}] "
            f"{query}"
        )

        runs = []

        for run_index in range(
            1,
            BENCHMARK_RUNS + 1,
        ):

            try:

                result = await run_query(
                    pipeline,
                    validator,
                    pii_masker,
                    regex_guard,
                    nemo_guard,
                    query,
                )

                timings = result[
                    "timings"
                ]

                runs.append(timings)

                print(
                    f"  Run {run_index}: "
                    f"{timings['total']:.4f}s "
                    f"({result['status']})"
                )

            except Exception:

                logger.exception(
                    "Benchmark failed for query: %s",
                    query,
                )

        # ----------------------------------------------------------
        # Per-query metrics
        # ----------------------------------------------------------

        component_metrics = {}

        components = [
            "input_validation",
            "pii_masking",
            "regex_guard",
            "nemo_guardrails",
            "dense_retrieval",
            "bm25_retrieval",
            "rrf",
            "reranking",
            "prompt_builder",
            "generation",
            "response_formatter",
            "total",
        ]

        for component in components:

            values = [
                run[component]
                for run in runs
                if component in run
            ]

            component_metrics[
                component
            ] = calculate_metrics(values)

        query_results.append(
            {
                "query": query,
                "runs": runs,
                "metrics": component_metrics,
            }
        )

    return {
        "configuration": {
            "query_count": len(
                BENCHMARK_QUERIES
            ),
            "runs_per_query": BENCHMARK_RUNS,
            "warmup_runs": WARMUP_RUNS,
        },
        "queries": query_results,
    }


# ------------------------------------------------------------------
# Overall metrics
# ------------------------------------------------------------------

def calculate_overall_metrics(
    query_results: dict,
) -> dict:

    component_values = {}

    components = [
        "input_validation",
        "pii_masking",
        "regex_guard",
        "nemo_guardrails",
        "dense_retrieval",
        "bm25_retrieval",
        "rrf",
        "reranking",
        "prompt_builder",
        "generation",
        "response_formatter",
        "total",
    ]

    for component in components:

        values = []

        for query in query_results[
            "queries"
        ]:

            for run in query["runs"]:

                if component in run:

                    values.append(
                        run[component]
                    )

        component_values[
            component
        ] = calculate_metrics(values)

    return component_values


# ------------------------------------------------------------------
# Console summary
# ------------------------------------------------------------------

def print_summary(
    startup_time: float,
    overall_metrics: dict,
) -> None:

    print()
    print("=" * 70)
    print("DOCUMIND BENCHMARK SUMMARY")
    print("=" * 70)

    print()
    print(
        f"Startup: "
        f"{startup_time:.4f}s"
    )

    print()
    print(
        f"{'Component':<25}"
        f"{'Mean':>12}"
        f"{'P50':>12}"
        f"{'P95':>12}"
        f"{'Max':>12}"
    )

    print("-" * 73)

    display_names = {
        "input_validation": "Input Validation",
        "pii_masking": "PII Masking",
        "regex_guard": "Regex Guard",
        "nemo_guardrails": "NeMo Guardrails",
        "dense_retrieval": "Dense Retrieval",
        "bm25_retrieval": "BM25 Retrieval",
        "rrf": "RRF",
        "reranking": "Reranking",
        "prompt_builder": "Prompt Builder",
        "generation": "Generation",
        "response_formatter": "Response Formatter",
        "total": "TOTAL",
    }

    for key, name in display_names.items():

        metrics = overall_metrics[key]

        print(
            f"{name:<25}"
            f"{metrics['mean']:>12.4f}"
            f"{metrics['p50']:>12.4f}"
            f"{metrics['p95']:>12.4f}"
            f"{metrics['max']:>12.4f}"
        )

    print("=" * 70)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

async def main():

    print()
    print("=" * 70)
    print("DOCUMIND PERFORMANCE BENCHMARK")
    print("=" * 70)

    # --------------------------------------------------------------
    # Startup
    # --------------------------------------------------------------

    startup = benchmark_startup()

    pipeline = startup["pipeline"]
    validator = startup["validator"]
    pii_masker = startup["pii_masker"]
    regex_guard = startup["regex_guard"]
    nemo_guard = startup["nemo_guard"]

    # --------------------------------------------------------------
    # Queries
    # --------------------------------------------------------------

    query_results = await benchmark_queries(
        pipeline,
        validator,
        pii_masker,
        regex_guard,
        nemo_guard,
    )

    # --------------------------------------------------------------
    # Overall
    # --------------------------------------------------------------

    overall_metrics = (
        calculate_overall_metrics(
            query_results
        )
    )

    # --------------------------------------------------------------
    # Save results
    # --------------------------------------------------------------

    save_json(
        "startup.json",
        {
            "startup_time_seconds": round(
                startup["startup_time"],
                6,
            )
        },
    )

    save_json(
        "query_latency.json",
        query_results,
    )

    save_json(
        "summary.json",
        {
            "startup_time_seconds": round(
                startup["startup_time"],
                6,
            ),
            "overall": overall_metrics,
        },
    )

    # --------------------------------------------------------------
    # Console output
    # --------------------------------------------------------------

    print_summary(
        startup_time=startup[
            "startup_time"
        ],
        overall_metrics=overall_metrics,
    )


if __name__ == "__main__":
    asyncio.run(main())