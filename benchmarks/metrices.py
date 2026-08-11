from statistics import mean, median
from typing import List


def percentile(
    values: List[float],
    percentile_value: float,
) -> float:
    """
    Calculate percentile using linear interpolation.

    Args:
        values: List of measured values.
        percentile_value: Percentile to calculate (0-100).

    Returns:
        Calculated percentile.
    """

    if not values:
        return 0.0

    sorted_values = sorted(values)

    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (
        len(sorted_values) - 1
    ) * (percentile_value / 100)

    lower = int(position)
    upper = min(
        lower + 1,
        len(sorted_values) - 1,
    )

    weight = position - lower

    return (
        sorted_values[lower]
        + (
            sorted_values[upper]
            - sorted_values[lower]
        )
        * weight
    )

def calculate_metrics(
    values: List[float],
) -> dict:
    """
    Calculate latency statistics.

    Args:
        values: Measured latency values in seconds.

    Returns:
        Dictionary containing benchmark metrics.
    """

    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    return {
        "count": len(values),
        "mean": round(mean(values), 6),
        "median": round(median(values), 6),
        "p50": round(
            percentile(values, 50),
            6,
        ),
        "p95": round(
            percentile(values, 95),
            6,
        ),
        "p99": round(
            percentile(values, 99),
            6,
        ),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }