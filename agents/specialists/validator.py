"""Validation specialist agent with tools."""
import re
import json
from typing import List, Dict, Any
from pydantic_ai import RunContext
from models.schemas import ValidationOutput
from models.dependencies import ValidatorDependencies
from agents.base import create_fast_agent


# Create the validation agent (using fast model for quick validation)
validator_agent = create_fast_agent(
    agent_id="validator",
    instructions="""
    You are a validation specialist focused on quality assurance.

    Your responsibilities:
    1. Validate content against specified criteria
    2. Identify issues and problems
    3. Provide actionable suggestions for improvement
    4. Assign quality scores (0.0 to 1.0)
    5. Be thorough but concise

    Guidelines:
    - Check all specified criteria carefully
    - Be specific about issues found
    - Provide constructive suggestions
    - Use strict mode when required for critical validations
    - Be honest about limitations

    Available tools:
    - check_format: Validate data format (JSON, XML, etc.)
    - check_length: Validate length constraints
    - check_patterns: Validate against regex patterns
    - check_completeness: Validate required fields
    """,
    deps_type=ValidatorDependencies,
    output_type=ValidationOutput
)


@validator_agent.tool
async def check_format(
    ctx: RunContext[ValidatorDependencies],
    content: str,
    format_type: str
) -> Dict[str, Any]:
    """
    Validate content format.

    Args:
        ctx: Agent context
        content: Content to validate
        format_type: Expected format (json, xml, email, url, etc.)

    Returns:
        Validation results
    """
    issues = []
    is_valid = True

    if format_type == "json":
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            is_valid = False
            issues.append(f"Invalid JSON: {str(e)}")

    elif format_type == "email":
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, content.strip()):
            is_valid = False
            issues.append("Invalid email format")

    elif format_type == "url":
        url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        if not re.match(url_pattern, content.strip()):
            is_valid = False
            issues.append("Invalid URL format")

    elif format_type == "phone":
        # Basic phone validation (can be improved)
        phone_pattern = r'^\+?1?\d{9,15}$'
        cleaned = re.sub(r'[\s\-\(\)]', '', content)
        if not re.match(phone_pattern, cleaned):
            is_valid = False
            issues.append("Invalid phone number format")

    else:
        issues.append(f"Unknown format type: {format_type}")
        is_valid = False

    return {
        "is_valid": is_valid,
        "format": format_type,
        "issues": issues
    }


@validator_agent.tool
async def check_length(
    ctx: RunContext[ValidatorDependencies],
    content: str,
    min_length: int = 0,
    max_length: int = None
) -> Dict[str, Any]:
    """
    Validate content length constraints.

    Args:
        ctx: Agent context
        content: Content to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length (None for no limit)

    Returns:
        Validation results
    """
    length = len(content)
    issues = []
    is_valid = True

    if length < min_length:
        is_valid = False
        issues.append(f"Content too short: {length} < {min_length}")

    if max_length is not None and length > max_length:
        is_valid = False
        issues.append(f"Content too long: {length} > {max_length}")

    return {
        "is_valid": is_valid,
        "actual_length": length,
        "min_length": min_length,
        "max_length": max_length,
        "issues": issues
    }


@validator_agent.tool
async def check_patterns(
    ctx: RunContext[ValidatorDependencies],
    content: str,
    patterns: List[str],
    match_all: bool = True
) -> Dict[str, Any]:
    """
    Validate content against regex patterns.

    Args:
        ctx: Agent context
        content: Content to validate
        patterns: List of regex patterns
        match_all: If True, all patterns must match; if False, at least one must match

    Returns:
        Validation results
    """
    matches = []
    issues = []

    for pattern in patterns:
        try:
            if re.search(pattern, content):
                matches.append(pattern)
            else:
                issues.append(f"Pattern not found: {pattern}")
        except re.error as e:
            issues.append(f"Invalid regex pattern '{pattern}': {str(e)}")

    if match_all:
        is_valid = len(matches) == len(patterns) and not issues
    else:
        is_valid = len(matches) > 0

    return {
        "is_valid": is_valid,
        "matched_patterns": matches,
        "total_patterns": len(patterns),
        "issues": issues
    }


@validator_agent.tool
async def check_completeness(
    ctx: RunContext[ValidatorDependencies],
    data: Dict[str, Any],
    required_fields: List[str]
) -> Dict[str, Any]:
    """
    Validate that required fields are present and non-empty.

    Args:
        ctx: Agent context
        data: Data dictionary to validate
        required_fields: List of required field names

    Returns:
        Validation results
    """
    issues = []
    missing_fields = []
    empty_fields = []

    for field in required_fields:
        if field not in data:
            missing_fields.append(field)
            issues.append(f"Missing required field: {field}")
        elif not data[field]:
            empty_fields.append(field)
            issues.append(f"Required field is empty: {field}")

    is_valid = len(issues) == 0

    return {
        "is_valid": is_valid,
        "missing_fields": missing_fields,
        "empty_fields": empty_fields,
        "total_required": len(required_fields),
        "issues": issues
    }


@validator_agent.tool
async def check_data_types(
    ctx: RunContext[ValidatorDependencies],
    data: Dict[str, Any],
    expected_types: Dict[str, str]
) -> Dict[str, Any]:
    """
    Validate data types of fields.

    Args:
        ctx: Agent context
        data: Data dictionary to validate
        expected_types: Dictionary of field_name -> expected_type

    Returns:
        Validation results
    """
    issues = []
    type_mismatches = []

    type_map = {
        "string": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict
    }

    for field, expected_type in expected_types.items():
        if field not in data:
            continue

        expected_class = type_map.get(expected_type)
        if expected_class is None:
            issues.append(f"Unknown expected type: {expected_type}")
            continue

        if not isinstance(data[field], expected_class):
            type_mismatches.append({
                "field": field,
                "expected": expected_type,
                "actual": type(data[field]).__name__
            })
            issues.append(
                f"Type mismatch for '{field}': expected {expected_type}, "
                f"got {type(data[field]).__name__}"
            )

    is_valid = len(issues) == 0

    return {
        "is_valid": is_valid,
        "type_mismatches": type_mismatches,
        "issues": issues
    }


@validator_agent.tool
async def validate_against_schema(
    ctx: RunContext[ValidatorDependencies],
    data: Dict[str, Any],
    schema: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate data against a schema definition.

    Args:
        ctx: Agent context
        data: Data to validate
        schema: Schema definition with required_fields, field_types, etc.

    Returns:
        Comprehensive validation results
    """
    all_issues = []
    results = {}

    # Check required fields
    if "required_fields" in schema:
        result = await check_completeness(ctx, data, schema["required_fields"])
        results["completeness"] = result
        all_issues.extend(result["issues"])

    # Check field types
    if "field_types" in schema:
        result = await check_data_types(ctx, data, schema["field_types"])
        results["types"] = result
        all_issues.extend(result["issues"])

    is_valid = len(all_issues) == 0

    return {
        "is_valid": is_valid,
        "validation_results": results,
        "total_issues": len(all_issues),
        "issues": all_issues
    }
