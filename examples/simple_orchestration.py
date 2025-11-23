"""Simple orchestration example."""
import asyncio
import httpx
from a2a.client import A2AClient


async def main():
    """
    Simple example demonstrating basic orchestration.

    This sends a task to the orchestrator which delegates to specialists.
    """
    print("🚀 Simple A2A Orchestration Example\n")

    # Connect to orchestrator
    async with A2AClient("http://localhost:8000") as client:
        print("📊 Sending task to orchestrator...")

        # Send a complex task that requires multiple specialists
        response = await client.send_message(
            message="""
            Analyze the following dataset and then generate Python code to visualize it:

            Dataset: [10, 25, 30, 45, 50, 55, 60, 75, 80, 90]

            Tasks:
            1. Calculate mean, median, and standard deviation
            2. Identify any trends in the data
            3. Generate Python code using matplotlib to create a line plot
            4. Validate that the code is syntactically correct
            """
        )

        # Extract result
        result = response.get("result", {})

        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return

        output = result.get("output", {})

        print("\n✅ Orchestration Complete!\n")
        print("=" * 80)

        # Display task results
        if "task_results" in output:
            print("\n📋 Task Results from Specialists:")
            for agent_name, task_result in output["task_results"].items():
                print(f"\n  {agent_name.upper()}:")
                print(f"    Success: {task_result.get('success')}")
                print(f"    Time: {task_result.get('execution_time', 0):.2f}s")

        # Display synthesis
        if "synthesis" in output:
            print(f"\n💡 Synthesis:\n{output['synthesis']}")

        # Display next actions
        if "next_actions" in output:
            print(f"\n🎯 Next Actions:")
            for action in output["next_actions"]:
                print(f"  - {action}")

        # Display metadata
        metadata = result.get("metadata", {})
        print(f"\n📊 Execution Strategy: {output.get('execution_strategy', 'N/A')}")
        print(f"⏱️  Total Time: {output.get('total_time', 0):.2f}s")
        print(f"🤖 Model: {metadata.get('model', 'N/A')}")

        print("\n" + "=" * 80)


async def test_individual_agent():
    """Test connecting to an individual specialist agent."""
    print("\n🧪 Testing Individual Agent (Analyst)\n")

    async with A2AClient("http://localhost:8001") as client:
        # Get agent metadata
        metadata = await client.get_agent_metadata()
        print(f"Agent: {metadata.get('name')}")
        print(f"Tools: {', '.join(metadata.get('capabilities', {}).get('tools', []))}")

        # Send simple analysis task
        response = await client.send_message(
            message="Calculate mean and median of: [5, 10, 15, 20, 25]"
        )

        result = response.get("result", {})
        output = result.get("output", {})

        if "error" not in result:
            print(f"\n✅ Analysis Result:")
            print(f"  Insights: {output.get('insights', [])[:2]}")
            print(f"  Metrics: {output.get('metrics', {})}")
            print(f"  Confidence: {output.get('confidence_score', 0):.2f}")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║        A2A Multi-Agent Orchestration - Simple Example       ║
╚══════════════════════════════════════════════════════════════╝

Prerequisites:
  1. Ollama is running (ollama serve)
  2. Models are pulled (llama3.1:8b, qwen2.5:7b, etc.)
  3. All agent servers are running:
     - Terminal 1: uvicorn main:orchestrator_app --port 8000
     - Terminal 2: uvicorn main:analyst_app --port 8001
     - Terminal 3: uvicorn main:coder_app --port 8002
     - Terminal 4: uvicorn main:validator_app --port 8003

""")

    try:
        asyncio.run(main())
        asyncio.run(test_individual_agent())
    except httpx.ConnectError:
        print("❌ Error: Could not connect to agents. Make sure all servers are running.")
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
