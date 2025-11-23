"""Data analyst specialist agent with tools."""
import statistics
from typing import List, Dict, Any
from pydantic_ai import RunContext
from models.schemas import AnalysisOutput
from models.dependencies import AnalystDependencies
from agents.base import create_analytical_agent


# Create the data analyst agent
data_analyst_agent = create_analytical_agent(
    agent_id="data-analyst",
    instructions="""
    You are a data analysis specialist with strong analytical capabilities.

    Your responsibilities:
    1. Analyze datasets and identify meaningful patterns
    2. Calculate relevant statistical metrics
    3. Provide actionable insights based on data
    4. Assign confidence scores to your findings (0.0 to 1.0)
    5. Explain your reasoning process clearly

    Guidelines:
    - Always validate data quality before analysis
    - Be concise and focus on the most important findings
    - Use the available tools to perform calculations
    - Provide reasoning for your insights
    - Be honest about uncertainty in your confidence scores

    Available tools:
    - calculate_statistics: Calculate statistical metrics on numeric data
    - analyze_trends: Identify trends in time-series data
    - compare_datasets: Compare two datasets and find differences
    """,
    deps_type=AnalystDependencies,
    output_type=AnalysisOutput
)


@data_analyst_agent.tool
async def calculate_statistics(
    ctx: RunContext[AnalystDependencies],
    data: List[float],
    metrics: List[str]
) -> Dict[str, float]:
    """
    Calculate statistical metrics on numeric data.

    Args:
        ctx: Agent context with dependencies
        data: List of numeric values to analyze
        metrics: List of metrics to calculate (mean, median, stdev, min, max, sum)

    Returns:
        Dictionary of calculated metrics
    """
    if not data:
        return {"error": "No data provided"}

    results = {}

    if 'mean' in metrics:
        results['mean'] = statistics.mean(data)

    if 'median' in metrics:
        results['median'] = statistics.median(data)

    if 'stdev' in metrics and len(data) > 1:
        results['stdev'] = statistics.stdev(data)

    if 'variance' in metrics and len(data) > 1:
        results['variance'] = statistics.variance(data)

    if 'min' in metrics:
        results['min'] = min(data)

    if 'max' in metrics:
        results['max'] = max(data)

    if 'sum' in metrics:
        results['sum'] = sum(data)

    if 'count' in metrics:
        results['count'] = len(data)

    return results


@data_analyst_agent.tool
async def analyze_trends(
    ctx: RunContext[AnalystDependencies],
    time_series: List[float],
    window_size: int = 3
) -> Dict[str, Any]:
    """
    Identify trends in time-series data using moving averages.

    Args:
        ctx: Agent context
        time_series: Sequential numeric values
        window_size: Window size for moving average

    Returns:
        Trend analysis results
    """
    if len(time_series) < window_size:
        return {"error": "Insufficient data for trend analysis"}

    # Calculate moving average
    moving_avg = []
    for i in range(len(time_series) - window_size + 1):
        window = time_series[i:i + window_size]
        moving_avg.append(sum(window) / window_size)

    # Determine overall trend
    if len(moving_avg) >= 2:
        first_half = sum(moving_avg[:len(moving_avg)//2]) / (len(moving_avg)//2)
        second_half = sum(moving_avg[len(moving_avg)//2:]) / (len(moving_avg) - len(moving_avg)//2)

        trend_direction = "increasing" if second_half > first_half else "decreasing"
        trend_strength = abs(second_half - first_half) / first_half if first_half != 0 else 0
    else:
        trend_direction = "unknown"
        trend_strength = 0

    return {
        "trend_direction": trend_direction,
        "trend_strength": round(trend_strength, 4),
        "moving_average": moving_avg,
        "latest_value": time_series[-1],
        "change_from_start": time_series[-1] - time_series[0]
    }


@data_analyst_agent.tool
async def compare_datasets(
    ctx: RunContext[AnalystDependencies],
    dataset_a: List[float],
    dataset_b: List[float],
    comparison_type: str = "basic"
) -> Dict[str, Any]:
    """
    Compare two datasets and identify differences.

    Args:
        ctx: Agent context
        dataset_a: First dataset
        dataset_b: Second dataset
        comparison_type: Type of comparison (basic, detailed)

    Returns:
        Comparison results
    """
    if not dataset_a or not dataset_b:
        return {"error": "Both datasets must have values"}

    results = {
        "dataset_a_size": len(dataset_a),
        "dataset_b_size": len(dataset_b),
        "dataset_a_mean": statistics.mean(dataset_a),
        "dataset_b_mean": statistics.mean(dataset_b),
    }

    # Calculate difference
    mean_diff = results["dataset_b_mean"] - results["dataset_a_mean"]
    results["mean_difference"] = mean_diff
    results["percent_change"] = (mean_diff / results["dataset_a_mean"] * 100) if results["dataset_a_mean"] != 0 else 0

    if comparison_type == "detailed":
        results["dataset_a_median"] = statistics.median(dataset_a)
        results["dataset_b_median"] = statistics.median(dataset_b)

        if len(dataset_a) > 1:
            results["dataset_a_stdev"] = statistics.stdev(dataset_a)
        if len(dataset_b) > 1:
            results["dataset_b_stdev"] = statistics.stdev(dataset_b)

    return results


@data_analyst_agent.tool
async def cache_result(
    ctx: RunContext[AnalystDependencies],
    key: str,
    value: Any
) -> bool:
    """
    Cache a result for future use.

    Args:
        ctx: Agent context
        key: Cache key
        value: Value to cache

    Returns:
        Success status
    """
    if not ctx.deps.cache_enabled:
        return False

    await ctx.deps.storage.set(f"analyst:cache:{key}", value, ttl=3600)
    return True


@data_analyst_agent.tool
async def get_cached_result(
    ctx: RunContext[AnalystDependencies],
    key: str
) -> Any:
    """
    Retrieve a cached result.

    Args:
        ctx: Agent context
        key: Cache key

    Returns:
        Cached value or None
    """
    if not ctx.deps.cache_enabled:
        return None

    return await ctx.deps.storage.get(f"analyst:cache:{key}")
