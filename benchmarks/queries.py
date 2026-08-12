"""
Benchmark query configuration.

The benchmark intentionally reuses the same query set used by
the main DocuMind pipeline tests so that performance measurements
are comparable with existing runs.
"""

from src.test.queries import main_pipeline_test_queries


BENCHMARK_QUERIES = list(main_pipeline_test_queries)


# Number of times each query is executed.
BENCHMARK_RUNS = 3

# Number of warm-up executions before collecting measurements.
WARMUP_RUNS = 1