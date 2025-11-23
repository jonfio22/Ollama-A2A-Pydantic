"""Base agent utilities and factory functions."""
from typing import TypeVar, Type
from pydantic import BaseModel
from pydantic_ai import Agent

DepsT = TypeVar('DepsT')
OutputT = TypeVar('OutputT', bound=BaseModel)


def create_agent(
    model: str,
    agent_id: str,
    instructions: str,
    deps_type: Type[DepsT],
    output_type: Type[OutputT],
    enable_instrumentation: bool = True
) -> Agent[DepsT, OutputT]:
    """
    Factory function to create standardized Pydantic AI agents.

    Args:
        model: Model identifier (e.g., 'ollama:llama3.1:8b')
        agent_id: Unique identifier for this agent
        instructions: System instructions for the agent
        deps_type: Type of dependencies for this agent
        output_type: Expected output type (Pydantic model)
        enable_instrumentation: Whether to enable Logfire instrumentation

    Returns:
        Configured Pydantic AI Agent instance
    """
    agent = Agent(
        model=model,
        deps_type=deps_type,
        result_type=output_type,
        system_prompt=instructions,
    )

    return agent


def create_fast_agent(
    agent_id: str,
    instructions: str,
    deps_type: Type[DepsT],
    output_type: Type[OutputT]
) -> Agent[DepsT, OutputT]:
    """
    Create a fast agent using llama3.2:3b for quick tasks.

    Args:
        agent_id: Unique identifier for this agent
        instructions: System instructions
        deps_type: Dependencies type
        output_type: Output type

    Returns:
        Fast-configured agent
    """
    return create_agent(
        model='ollama:llama3.2:3b',
        agent_id=agent_id,
        instructions=instructions,
        deps_type=deps_type,
        output_type=output_type
    )


def create_analytical_agent(
    agent_id: str,
    instructions: str,
    deps_type: Type[DepsT],
    output_type: Type[OutputT]
) -> Agent[DepsT, OutputT]:
    """
    Create an analytical agent using qwen2.5:7b for data analysis.

    Args:
        agent_id: Unique identifier for this agent
        instructions: System instructions
        deps_type: Dependencies type
        output_type: Output type

    Returns:
        Analytical agent
    """
    return create_agent(
        model='ollama:qwen2.5:7b',
        agent_id=agent_id,
        instructions=instructions,
        deps_type=deps_type,
        output_type=output_type
    )


def create_coding_agent(
    agent_id: str,
    instructions: str,
    deps_type: Type[DepsT],
    output_type: Type[OutputT]
) -> Agent[DepsT, OutputT]:
    """
    Create a coding agent using deepseek-coder-v2:16b for code generation.

    Args:
        agent_id: Unique identifier for this agent
        instructions: System instructions
        deps_type: Dependencies type
        output_type: Output type

    Returns:
        Coding specialist agent
    """
    return create_agent(
        model='ollama:deepseek-coder-v2:16b',
        agent_id=agent_id,
        instructions=instructions,
        deps_type=deps_type,
        output_type=output_type
    )


def create_orchestrator_agent(
    agent_id: str,
    instructions: str,
    deps_type: Type[DepsT],
    output_type: Type[OutputT]
) -> Agent[DepsT, OutputT]:
    """
    Create an orchestrator agent using llama3.1:8b for coordination.

    Args:
        agent_id: Unique identifier for this agent
        instructions: System instructions
        deps_type: Dependencies type
        output_type: Output type

    Returns:
        Orchestrator agent
    """
    return create_agent(
        model='ollama:llama3.1:8b',
        agent_id=agent_id,
        instructions=instructions,
        deps_type=deps_type,
        output_type=output_type
    )
