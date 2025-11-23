"""Parallel execution example."""
import asyncio
import time
from a2a.client import A2AClient


async def main():
    """
    Demonstrate parallel execution of multiple agents.

    This example shows how the orchestrator can delegate to multiple
    specialists concurrently for faster execution.
    """
    print("⚡ Parallel Execution Example\n")

    async with A2AClient("http://localhost:8000") as client:
        start_time = time.time()

        print("📤 Sending parallel tasks to orchestrator...")

        # Task that should trigger parallel execution
        response = await client.send_message(
            message="""
            I need you to perform three independent tasks in parallel:

            1. ANALYST: Analyze this dataset for trends: [100, 105, 110, 108, 115, 120, 125, 130, 128, 135]

            2. CODER: Generate a Python function that calculates the Fibonacci sequence up to n terms

            3. VALIDATOR: Validate this email address: user@example.com

            These tasks are independent and can run concurrently.
            """
        )

        execution_time = time.time() - start_time

        result = response.get("result", {})
        output = result.get("output", {})

        print("\n✅ All Tasks Complete!\n")
        print("=" * 80)

        # Display results from each specialist
        if "task_results" in output:
            print("\n📋 Results from Parallel Execution:\n")

            for agent_name, task_result in output["task_results"].items():
                agent_output = task_result.get("output", {}).get("output", {})
                exec_time = task_result.get("execution_time", 0)

                print(f"\n  🤖 {agent_name.upper()}")
                print(f"  ⏱️  Execution Time: {exec_time:.2f}s")
                print(f"  ✓  Success: {task_result.get('success')}")

                # Display agent-specific output
                if agent_name == "analyst" and "insights" in agent_output:
                    print(f"  💡 Insights: {agent_output['insights'][:1]}")
                elif agent_name == "coder" and "code" in agent_output:
                    print(f"  💻 Code Generated: {len(agent_output.get('code', ''))} characters")
                elif agent_name == "validator" and "is_valid" in agent_output:
                    print(f"  ✅ Validation: {agent_output['is_valid']}")

        # Show execution strategy used
        strategy = output.get("execution_strategy", "unknown")
        print(f"\n🎯 Execution Strategy: {strategy}")

        # Compare total time vs sum of individual times
        total_agent_time = sum(
            task["execution_time"]
            for task in output.get("task_results", {}).values()
        )

        print(f"\n⏱️  Performance Metrics:")
        print(f"  Total Wall Time: {execution_time:.2f}s")
        print(f"  Sum of Agent Times: {total_agent_time:.2f}s")
        print(f"  Speedup from Parallelization: {total_agent_time / execution_time:.2f}x")

        print("\n" + "=" * 80)


async def compare_sequential_vs_parallel():
    """
    Compare sequential vs parallel execution times.
    """
    print("\n📊 Sequential vs Parallel Comparison\n")

    async with A2AClient("http://localhost:8000") as client:
        # Sequential execution
        print("1️⃣  Running SEQUENTIAL execution...")
        seq_start = time.time()

        await client.send_message(
            message="""
            Perform these tasks one after another:
            1. First analyze: [1, 2, 3, 4, 5]
            2. Then generate code for a hello world function
            3. Finally validate this data
            """
        )

        seq_time = time.time() - seq_start
        print(f"   ⏱️  Sequential time: {seq_time:.2f}s")

        # Parallel execution
        print("\n2️⃣  Running PARALLEL execution...")
        par_start = time.time()

        await client.send_message(
            message="""
            Perform these independent tasks in parallel:
            1. ANALYST: analyze [1, 2, 3, 4, 5]
            2. CODER: generate a hello world function
            3. VALIDATOR: validate some data

            These can run concurrently.
            """
        )

        par_time = time.time() - par_start
        print(f"   ⏱️  Parallel time: {par_time:.2f}s")

        # Show improvement
        improvement = ((seq_time - par_time) / seq_time * 100)
        print(f"\n🚀 Performance Improvement: {improvement:.1f}% faster")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║      A2A Multi-Agent Orchestration - Parallel Execution     ║
╚══════════════════════════════════════════════════════════════╝

This example demonstrates the power of parallel agent execution.
When tasks are independent, they run concurrently for faster results.

""")

    try:
        asyncio.run(main())
        asyncio.run(compare_sequential_vs_parallel())
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure all agent servers are running!")
