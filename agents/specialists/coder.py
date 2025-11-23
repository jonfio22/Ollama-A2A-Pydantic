"""Code generation specialist agent with tools."""
import re
from typing import List, Dict, Any
from pydantic_ai import RunContext
from models.schemas import CodeOutput
from models.dependencies import CoderDependencies
from agents.base import create_coding_agent


# Create the code generation agent
code_generator_agent = create_coding_agent(
    agent_id="code-generator",
    instructions="""
    You are a code generation specialist with expertise in multiple programming languages.

    Your responsibilities:
    1. Generate clean, well-structured code based on requirements
    2. Provide clear explanations of the code
    3. Include test cases when requested
    4. Identify and list dependencies
    5. Follow best practices and coding standards

    Guidelines:
    - Write production-quality code with proper error handling
    - Use meaningful variable and function names
    - Add comments for complex logic
    - Consider edge cases and validation
    - Be honest about confidence in your solution

    Available tools:
    - validate_syntax: Check code syntax
    - generate_tests: Generate unit tests for code
    - find_dependencies: Extract dependencies from code
    - format_code: Format code according to standards
    """,
    deps_type=CoderDependencies,
    output_type=CodeOutput
)


@code_generator_agent.tool
async def validate_syntax(
    ctx: RunContext[CoderDependencies],
    code: str,
    language: str = "python"
) -> Dict[str, Any]:
    """
    Validate code syntax.

    Args:
        ctx: Agent context
        code: Code to validate
        language: Programming language

    Returns:
        Validation results
    """
    if language == "python":
        try:
            compile(code, '<string>', 'exec')
            return {"valid": True, "errors": []}
        except SyntaxError as e:
            return {
                "valid": False,
                "errors": [f"Line {e.lineno}: {e.msg}"]
            }
    else:
        # For other languages, do basic checks
        issues = []

        # Check for unmatched braces
        if code.count('{') != code.count('}'):
            issues.append("Unmatched braces")

        if code.count('(') != code.count(')'):
            issues.append("Unmatched parentheses")

        if code.count('[') != code.count(']'):
            issues.append("Unmatched brackets")

        return {
            "valid": len(issues) == 0,
            "errors": issues
        }


@code_generator_agent.tool
async def generate_tests(
    ctx: RunContext[CoderDependencies],
    function_code: str,
    language: str = "python"
) -> str:
    """
    Generate unit tests for a function.

    Args:
        ctx: Agent context
        function_code: Function code to test
        language: Programming language

    Returns:
        Generated test code
    """
    if language == "python":
        # Extract function name
        match = re.search(r'def\s+(\w+)\s*\(', function_code)
        if not match:
            return "# Could not extract function name"

        func_name = match.group(1)

        test_template = f"""
import pytest

def test_{func_name}_basic():
    \"\"\"Test basic functionality of {func_name}.\"\"\"
    # TODO: Add test implementation
    pass

def test_{func_name}_edge_cases():
    \"\"\"Test edge cases for {func_name}.\"\"\"
    # TODO: Add edge case tests
    pass

def test_{func_name}_error_handling():
    \"\"\"Test error handling in {func_name}.\"\"\"
    # TODO: Add error handling tests
    pass
"""
        return test_template

    return f"# Test generation not implemented for {language}"


@code_generator_agent.tool
async def find_dependencies(
    ctx: RunContext[CoderDependencies],
    code: str,
    language: str = "python"
) -> List[str]:
    """
    Extract dependencies from code.

    Args:
        ctx: Agent context
        code: Source code
        language: Programming language

    Returns:
        List of dependencies
    """
    dependencies = []

    if language == "python":
        # Find import statements
        import_pattern = r'^(?:from\s+(\S+)\s+)?import\s+(.+?)(?:\s+as\s+\S+)?$'
        for line in code.split('\n'):
            line = line.strip()
            match = re.match(import_pattern, line)
            if match:
                if match.group(1):  # from X import Y
                    module = match.group(1).split('.')[0]
                else:  # import X
                    imports = match.group(2).split(',')
                    module = imports[0].strip().split('.')[0]

                # Skip standard library (basic check)
                if module not in ['os', 'sys', 'json', 're', 'time', 'datetime',
                                 'collections', 'itertools', 'functools']:
                    if module not in dependencies:
                        dependencies.append(module)

    return dependencies


@code_generator_agent.tool
async def format_code(
    ctx: RunContext[CoderDependencies],
    code: str,
    language: str = "python"
) -> str:
    """
    Format code according to language standards.

    Args:
        ctx: Agent context
        code: Code to format
        language: Programming language

    Returns:
        Formatted code
    """
    # Basic formatting (in production, would use actual formatters like black, prettier, etc.)

    if language == "python":
        # Remove trailing whitespace
        lines = code.split('\n')
        lines = [line.rstrip() for line in lines]

        # Ensure blank line at end
        if lines and lines[-1]:
            lines.append('')

        return '\n'.join(lines)

    return code


@code_generator_agent.tool
async def save_code(
    ctx: RunContext[CoderDependencies],
    code: str,
    filename: str
) -> bool:
    """
    Save generated code to storage.

    Args:
        ctx: Agent context
        code: Code to save
        filename: Filename

    Returns:
        Success status
    """
    await ctx.deps.storage.set(
        f"coder:generated:{filename}",
        code,
        ttl=86400  # 24 hours
    )
    return True


@code_generator_agent.tool
async def get_code_template(
    ctx: RunContext[CoderDependencies],
    template_type: str,
    language: str = "python"
) -> str:
    """
    Get a code template for common patterns.

    Args:
        ctx: Agent context
        template_type: Type of template (class, function, api_endpoint, etc.)
        language: Programming language

    Returns:
        Code template
    """
    templates = {
        "python": {
            "function": '''def function_name(param1, param2):
    """
    Brief description.

    Args:
        param1: Description
        param2: Description

    Returns:
        Description
    """
    # Implementation
    pass
''',
            "class": '''class ClassName:
    """Brief description of the class."""

    def __init__(self, param1):
        """Initialize the class."""
        self.param1 = param1

    def method_name(self):
        """Brief description of method."""
        pass
''',
            "api_endpoint": '''from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class RequestModel(BaseModel):
    """Request model."""
    field: str

class ResponseModel(BaseModel):
    """Response model."""
    result: str

@router.post("/endpoint", response_model=ResponseModel)
async def endpoint_name(request: RequestModel):
    """Endpoint description."""
    # Implementation
    return ResponseModel(result="success")
'''
        }
    }

    return templates.get(language, {}).get(template_type, "# Template not found")
