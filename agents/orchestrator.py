"""Orchestrator agent for multi-agent coordination."""
import time
import asyncio
from typing import Dict, Any, List
from pydantic_ai import RunContext
from models.schemas import OrchestratorOutput, TaskResult
from models.dependencies import OrchestratorDependencies
from agents.base import create_orchestrator_agent


# Create the orchestrator agent
orchestrator_agent = create_orchestrator_agent(
    agent_id="orchestrator",
    instructions="""
    You are the orchestrator agent responsible for coordinating multiple specialist agents.

    Your responsibilities:
    1. Analyze incoming requests and determine which specialists to engage
    2. Decide between sequential or parallel execution based on task dependencies
    3. Delegate tasks to appropriate specialist agents via A2A protocol
    4. Aggregate and synthesize results from multiple agents
    5. Provide comprehensive final output with next steps

    Available specialists:
    - analyst: Data analysis, statistics, trend identification (strong analytical reasoning)
    - coder: Code generation, testing, syntax validation (specialized for coding)
    - validator: Quality assurance, format validation, completeness checks (fast validation)

    Decision guidelines:
    - Use parallel execution when tasks are independent
    - Use sequential execution when later tasks depend on earlier results
    - Only delegate to specialists when their expertise is truly needed
    - Be efficient and avoid unnecessary agent calls
    - Provide clear synthesis of results from all agents

    Tools available:
    - delegate_to_specialist: Send task to a specialist agent
    - delegate_parallel: Send tasks to multiple agents concurrently
    - get_specialist_capabilities: Get metadata about a specialist
    - save_intermediate_result: Cache intermediate results
    """,
    deps_type=OrchestratorDependencies,
    output_type=OrchestratorOutput
)


@orchestrator_agent.tool
async def delegate_to_specialist(
    ctx: RunContext[OrchestratorDependencies],
    specialist_name: str,
    task_description: str,
    context_data: Dict[str, Any] = None
) -> TaskResult:
    """
    Send a task to a specialist agent via A2A protocol.

    Args:
        ctx: Agent context with dependencies
        specialist_name: Name of the specialist (analyst, coder, validator)
        task_description: Description of the task to perform
        context_data: Additional context data to pass

    Returns:
        TaskResult with agent output and execution metadata
    """
    start_time = time.time()

    endpoint = ctx.deps.specialist_agents.get(specialist_name)
    if not endpoint:
        return TaskResult(
            agent=specialist_name,
            output={"error": f"Unknown specialist: {specialist_name}"},
            execution_time=0,
            success=False
        )

    # Prepare A2A protocol request
    payload = {
        "jsonrpc": "2.0",
        "id": f"{ctx.deps.agent_id}-{specialist_name}-{int(time.time())}",
        "method": "run",
        "params": {
            "message": task_description,
        }
    }

    if context_data:
        payload["params"]["artifacts"] = [context_data]

    try:
        response = await ctx.deps.http_client.post(
            f"{endpoint}/a2a/run",
            json=payload,
            timeout=60.0
        )
        response.raise_for_status()
        result = response.json()

        execution_time = time.time() - start_time

        return TaskResult(
            agent=specialist_name,
            output=result.get("result", {}),
            execution_time=execution_time,
            success=True
        )

    except Exception as e:
        execution_time = time.time() - start_time
        return TaskResult(
            agent=specialist_name,
            output={"error": str(e)},
            execution_time=execution_time,
            success=False
        )


@orchestrator_agent.tool
async def delegate_parallel(
    ctx: RunContext[OrchestratorDependencies],
    tasks: List[Dict[str, Any]]
) -> Dict[str, TaskResult]:
    """
    Send multiple tasks to different agents concurrently.

    Args:
        ctx: Agent context
        tasks: List of task dictionaries with 'specialist', 'task', and optional 'context'

    Returns:
        Dictionary mapping agent names to their results
    """
    # Create concurrent tasks
    async_tasks = []
    agent_names = []

    for task_spec in tasks:
        specialist = task_spec.get("specialist")
        task_desc = task_spec.get("task")
        context = task_spec.get("context")

        if not specialist or not task_desc:
            continue

        agent_names.append(specialist)
        async_tasks.append(
            delegate_to_specialist(ctx, specialist, task_desc, context)
        )

    # Execute all tasks concurrently
    results = await asyncio.gather(*async_tasks, return_exceptions=True)

    # Build results dictionary
    result_dict = {}
    for agent_name, result in zip(agent_names, results):
        if isinstance(result, Exception):
            result_dict[agent_name] = TaskResult(
                agent=agent_name,
                output={"error": str(result)},
                execution_time=0,
                success=False
            )
        else:
            result_dict[agent_name] = result

    return result_dict


@orchestrator_agent.tool
async def get_specialist_capabilities(
    ctx: RunContext[OrchestratorDependencies],
    specialist_name: str
) -> Dict[str, Any]:
    """
    Retrieve metadata about a specialist agent.

    Args:
        ctx: Agent context
        specialist_name: Name of the specialist

    Returns:
        Agent capabilities and metadata
    """
    endpoint = ctx.deps.specialist_agents.get(specialist_name)
    if not endpoint:
        return {"error": f"Unknown specialist: {specialist_name}"}

    try:
        response = await ctx.deps.http_client.get(
            f"{endpoint}/.well-known/agent.json",
            timeout=5.0
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


@orchestrator_agent.tool
async def save_intermediate_result(
    ctx: RunContext[OrchestratorDependencies],
    key: str,
    result: Any
) -> bool:
    """
    Save intermediate results for later retrieval.

    Args:
        ctx: Agent context
        key: Storage key
        result: Result to save

    Returns:
        Success status
    """
    if not ctx.deps.task_storage:
        return False

    try:
        # Store in simple storage if available
        await ctx.deps.task_storage.set(
            f"orchestrator:intermediate:{key}",
            result,
            ttl=3600  # 1 hour
        )
        return True
    except:
        return False


@orchestrator_agent.tool
async def get_intermediate_result(
    ctx: RunContext[OrchestratorDependencies],
    key: str
) -> Any:
    """
    Retrieve previously saved intermediate result.

    Args:
        ctx: Agent context
        key: Storage key

    Returns:
        Saved result or None
    """
    if not ctx.deps.task_storage:
        return None

    try:
        return await ctx.deps.task_storage.get(f"orchestrator:intermediate:{key}")
    except:
        return None


@orchestrator_agent.tool
async def analyze_task_complexity(
    ctx: RunContext[OrchestratorDependencies],
    task_description: str
) -> Dict[str, Any]:
    """
    Analyze the complexity of a task to determine execution strategy.

    Args:
        ctx: Agent context
        task_description: Description of the task

    Returns:
        Complexity analysis with recommended strategy
    """
    # Simple heuristic-based analysis
    words = task_description.lower().split()

    # Check for keywords indicating multiple sub-tasks
    multi_task_keywords = ['and', 'also', 'then', 'after', 'both', 'multiple']
    has_multiple_parts = any(keyword in words for keyword in multi_task_keywords)

    # Check for keywords indicating dependencies
    sequential_keywords = ['then', 'after', 'following', 'based on', 'using']
    has_dependencies = any(keyword in words for keyword in sequential_keywords)

    # Check for specialist keywords
    needs_analyst = any(word in words for word in ['analyze', 'data', 'statistics', 'trend'])
    needs_coder = any(word in words for word in ['code', 'program', 'function', 'script'])
    needs_validator = any(word in words for word in ['validate', 'check', 'verify', 'quality'])

    specialists_needed = []
    if needs_analyst:
        specialists_needed.append("analyst")
    if needs_coder:
        specialists_needed.append("coder")
    if needs_validator:
        specialists_needed.append("validator")

    # Determine recommended strategy
    if has_dependencies:
        strategy = "sequential"
    elif has_multiple_parts and len(specialists_needed) > 1:
        strategy = "parallel"
    else:
        strategy = "single"

    return {
        "has_multiple_parts": has_multiple_parts,
        "has_dependencies": has_dependencies,
        "specialists_needed": specialists_needed,
        "recommended_strategy": strategy,
        "estimated_complexity": "high" if len(specialists_needed) > 2 else "medium" if specialists_needed else "low"
    }
