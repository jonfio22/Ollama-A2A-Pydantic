"""Base agent utilities and factory functions."""
from typing import TypeVar, Type
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.output import PromptedOutput
from config.settings import settings

DepsT = TypeVar('DepsT')
OutputT = TypeVar('OutputT', bound=BaseModel)


def create_ollama_model(model_name: str, base_url: str = "http://localhost:11434/v1") -> OpenAIChatModel:
    """
    Create an Ollama model configuration for Pydantic AI.

    Args:
        model_name: Name of the Ollama model (e.g., 'llama3.1:8b')
        base_url: Base URL for Ollama server (default: http://localhost:11434/v1)

    Returns:
        Configured OpenAIChatModel with OllamaProvider
    """
    return OpenAIChatModel(
        model_name=model_name,
        provider=OllamaProvider(base_url=base_url),
    )


def create_agent(
    model: str,
    agent_id: str,
    instructions: str,
    deps_type: Type[DepsT],
    output_type: Type[OutputT],
    enable_instrumentation: bool = True,
    retries: int = 2,
    output_retries: int = 3
) -> Agent[DepsT, OutputT]:
    """
    Factory function to create standardized Pydantic AI agents.

    Args:
        model: Model identifier (e.g., 'ollama:llama3.1:8b' or just 'llama3.1:8b')
        agent_id: Unique identifier for this agent
        instructions: System instructions for the agent
        deps_type: Type of dependencies for this agent
        output_type: Expected output type (Pydantic model)
        enable_instrumentation: Whether to enable Logfire instrumentation
        retries: Number of retries for tool calls (default: 2)
        output_retries: Number of retries for output validation (default: 3)

    Returns:
        Configured Pydantic AI Agent instance
    """
    # Strip 'ollama:' prefix if present
    model_name = model.replace('ollama:', '')

    # Create Ollama model configuration
    ollama_model = create_ollama_model(model_name)

    # Use PromptedOutput for better Ollama compatibility
    # Ollama models work more reliably when prompted to output JSON
    # rather than using tool calling for structured output
    prompted_output = PromptedOutput(output_type)

    agent = Agent(
        model=ollama_model,
        deps_type=deps_type,
        output_type=prompted_output,
        system_prompt=instructions,
        retries=retries,
        output_retries=output_retries,
    )

    return agent


def create_fast_agent(
    agent_id: str,
    instructions: str,
    deps_type: Type[DepsT],
    output_type: Type[OutputT]
) -> Agent[DepsT, OutputT]:
    """
    Create a fast agent using the configured fast model for quick tasks.

    Args:
        agent_id: Unique identifier for this agent
        instructions: System instructions
        deps_type: Dependencies type
        output_type: Output type

    Returns:
        Fast-configured agent
    """
    return create_agent(
        model=settings.fast_model,
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
    Create an analytical agent using the configured analyst model for data analysis.

    Args:
        agent_id: Unique identifier for this agent
        instructions: System instructions
        deps_type: Dependencies type
        output_type: Output type

    Returns:
        Analytical agent
    """
    return create_agent(
        model=settings.analyst_model,
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
    Create a coding agent using the configured coder model for code generation.

    Note: Model is configurable via CODER_MODEL env var. Default uses deepseek-coder:6.7b
    for superior code generation performance.

    Args:
        agent_id: Unique identifier for this agent
        instructions: System instructions
        deps_type: Dependencies type
        output_type: Output type

    Returns:
        Coding specialist agent
    """
    return create_agent(
        model=settings.coder_model,
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
    Create an orchestrator agent using the configured orchestrator model for coordination.

    Note: Orchestrator needs more output retries due to complex structured output
    with nested models and multiple fields. Default uses phi3:3.8b for lightweight
    efficient reasoning with strong task decomposition (50% smaller than original).

    Args:
        agent_id: Unique identifier for this agent
        instructions: System instructions
        deps_type: Dependencies type
        output_type: Output type

    Returns:
        Orchestrator agent
    """
    return create_agent(
        model=settings.orchestrator_model,
        agent_id=agent_id,
        instructions=instructions,
        deps_type=deps_type,
        output_type=output_type,
        output_retries=8  # Higher retries for complex orchestration output
    )


def create_vision_agent(
    agent_id: str,
    instructions: str,
    deps_type: Type[DepsT],
    output_type: Type[OutputT]
) -> Agent[DepsT, OutputT]:
    """
    Create a vision agent using the configured vision model for multimodal tasks.

    Note: Vision inference can be more variable; uses extra retries for reliability.
    Default uses moondream:1.8b (lightweight multimodal model) for fast image understanding
    on resource-constrained systems. For higher quality, use minicpm-v:latest instead.

    Args:
        agent_id: Unique identifier for this agent
        instructions: System instructions
        deps_type: Dependencies type
        output_type: Output type

    Returns:
        Vision specialist agent
    """
    return create_agent(
        model=settings.vision_model,
        agent_id=agent_id,
        instructions=instructions,
        deps_type=deps_type,
        output_type=output_type,
        retries=3,  # Vision inference can be flakier
        output_retries=5  # More retries for multimodal output complexity
    )
