#!/usr/bin/env python3
"""Quick test script for A2A agents."""
import asyncio
import sys
import os

# Add current directory to path to find a2a module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from a2a.client import A2AClient


async def test_orchestrator():
    """Test the orchestrator with a simple task."""
    print("🤖 Testing Orchestrator Agent...\n")

    async with A2AClient("http://localhost:8000") as client:
        # Simple task that involves multiple specialists
        response = await client.send_message(
            message="""
            Analyze this dataset: [5, 10, 15, 20, 25, 30]

            Then write Python code to calculate the mean and plot it.
            """
        )

        result = response.get("result", {})

        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return False

        print("✅ Orchestrator Response:")
        output = result.get("output", {})
        print(f"   Strategy: {output.get('execution_strategy')}")
        print(f"   Total Time: {output.get('total_time', 0):.2f}s")

        if "task_results" in output:
            print(f"   Agents Used: {', '.join(output['task_results'].keys())}")

        return True


async def test_analyst():
    """Test the analyst agent directly."""
    print("\n📊 Testing Analyst Agent...\n")

    async with A2AClient("http://localhost:8001") as client:
        response = await client.send_message(
            message="Calculate the mean, median, and standard deviation of [12, 15, 18, 21, 24]"
        )

        result = response.get("result", {})

        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return False

        print("✅ Analyst Response:")
        output = result.get("output", {})
        print(f"   Metrics: {output.get('metrics', {})}")
        print(f"   Confidence: {output.get('confidence_score', 0):.2f}")

        return True


async def test_coder():
    """Test the coder agent directly."""
    print("\n💻 Testing Coder Agent...\n")

    async with A2AClient("http://localhost:8002") as client:
        response = await client.send_message(
            message="Write a Python function to calculate the factorial of a number"
        )

        result = response.get("result", {})

        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return False

        print("✅ Coder Response:")
        output = result.get("output", {})
        code = output.get("code", "")
        if code:
            print(f"   Generated {len(code)} characters of code")
            print(f"   Language: {output.get('language', 'unknown')}")

        return True


async def main():
    """Run all tests."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║              A2A Agent Quick Test Suite                     ║
╚══════════════════════════════════════════════════════════════╝
""")

    results = []

    # Test each agent
    try:
        results.append(("Orchestrator", await test_orchestrator()))
        results.append(("Analyst", await test_analyst()))
        results.append(("Coder", await test_coder()))
    except Exception as e:
        print(f"\n❌ Connection Error: {e}")
        print("\nMake sure all agents are running with: ./run.sh")
        sys.exit(1)

    # Summary
    print("\n" + "="*60)
    print("📋 Test Summary:")
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {name}: {status}")

    all_passed = all(r[1] for r in results)
    print("="*60)

    if all_passed:
        print("\n🎉 All agents are working correctly!\n")
    else:
        print("\n⚠️  Some agents failed. Check the output above.\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
