"""Unit tests for agents."""
import pytest
from models.dependencies import (
    AnalystDependencies,
    CoderDependencies,
    ValidatorDependencies
)
from storage.memory_storage import MemorySimpleStorage
from agents.specialists.analyst import data_analyst_agent
from agents.specialists.coder import code_generator_agent
from agents.specialists.validator import validator_agent


@pytest.mark.asyncio
async def test_analyst_agent():
    """Test data analyst agent with mock dependencies."""
    deps = AnalystDependencies(
        agent_id="test-analyst",
        storage=MemorySimpleStorage(),
        cache_enabled=True
    )

    result = await data_analyst_agent.run(
        "Calculate the mean and median of these numbers: [10, 20, 30, 40, 50]",
        deps=deps
    )

    assert result.data is not None
    assert hasattr(result.data, 'insights')
    assert hasattr(result.data, 'confidence_score')
    assert 0 <= result.data.confidence_score <= 1


@pytest.mark.asyncio
async def test_coder_agent():
    """Test code generator agent."""
    deps = CoderDependencies(
        agent_id="test-coder",
        storage=MemorySimpleStorage(),
        language="python",
        include_tests=True
    )

    result = await code_generator_agent.run(
        "Generate a Python function that adds two numbers",
        deps=deps
    )

    assert result.data is not None
    assert hasattr(result.data, 'code')
    assert hasattr(result.data, 'confidence')
    assert len(result.data.code) > 0


@pytest.mark.asyncio
async def test_validator_agent():
    """Test validator agent."""
    deps = ValidatorDependencies(
        agent_id="test-validator",
        storage=MemorySimpleStorage(),
        strict_mode=False
    )

    result = await validator_agent.run(
        "Validate this email: test@example.com",
        deps=deps
    )

    assert result.data is not None
    assert hasattr(result.data, 'is_valid')
    assert hasattr(result.data, 'score')
    assert isinstance(result.data.is_valid, bool)


@pytest.mark.asyncio
async def test_analyst_tool_calculate_statistics():
    """Test analyst's calculate_statistics tool."""
    from agents.specialists.analyst import calculate_statistics
    from pydantic_ai import RunContext

    deps = AnalystDependencies(
        agent_id="test",
        storage=MemorySimpleStorage(),
        cache_enabled=True
    )

    # Create mock context
    class MockContext:
        deps = deps

    ctx = MockContext()

    # Test statistics calculation
    result = await calculate_statistics(
        ctx,
        data=[10.0, 20.0, 30.0, 40.0, 50.0],
        metrics=['mean', 'median', 'min', 'max']
    )

    assert result['mean'] == 30.0
    assert result['median'] == 30.0
    assert result['min'] == 10.0
    assert result['max'] == 50.0


@pytest.mark.asyncio
async def test_coder_tool_validate_syntax():
    """Test coder's validate_syntax tool."""
    from agents.specialists.coder import validate_syntax

    deps = CoderDependencies(
        agent_id="test",
        storage=MemorySimpleStorage()
    )

    class MockContext:
        deps = deps

    ctx = MockContext()

    # Test valid Python code
    result = await validate_syntax(
        ctx,
        code="def hello():\n    print('hello')",
        language="python"
    )

    assert result['valid'] is True
    assert len(result['errors']) == 0

    # Test invalid Python code
    result = await validate_syntax(
        ctx,
        code="def hello(\n    print('hello')",
        language="python"
    )

    assert result['valid'] is False
    assert len(result['errors']) > 0


@pytest.mark.asyncio
async def test_validator_tool_check_format():
    """Test validator's check_format tool."""
    from agents.specialists.validator import check_format

    deps = ValidatorDependencies(
        agent_id="test",
        storage=MemorySimpleStorage()
    )

    class MockContext:
        deps = deps

    ctx = MockContext()

    # Test valid email
    result = await check_format(
        ctx,
        content="test@example.com",
        format_type="email"
    )

    assert result['is_valid'] is True
    assert len(result['issues']) == 0

    # Test invalid email
    result = await check_format(
        ctx,
        content="not-an-email",
        format_type="email"
    )

    assert result['is_valid'] is False
    assert len(result['issues']) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
