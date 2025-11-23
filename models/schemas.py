"""Pydantic models for agent inputs and outputs."""
from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field


# ============================================================================
# Enums
# ============================================================================

class TaskStatus(str, Enum):
    """Status of a task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentRole(str, Enum):
    """Role of an agent in the system."""
    ORCHESTRATOR = "orchestrator"
    ANALYST = "analyst"
    CODER = "coder"
    VALIDATOR = "validator"


# ============================================================================
# Base Models
# ============================================================================

class BaseAgentOutput(BaseModel):
    """Base output schema for all agents."""
    success: bool = Field(description="Whether the task succeeded")
    message: str = Field(description="Human-readable message")
    confidence: float = Field(ge=0, le=1, description="Confidence score 0-1")


# ============================================================================
# Data Analysis Models
# ============================================================================

class AnalysisRequest(BaseModel):
    """Request for data analysis."""
    query: str = Field(description="Analysis query")
    dataset: Optional[str] = Field(None, description="Dataset identifier")
    metrics: List[str] = Field(default_factory=list, description="Metrics to calculate")


class AnalysisOutput(BaseModel):
    """Output from data analysis agent."""
    insights: List[str] = Field(description="Key insights from analysis")
    metrics: Dict[str, float] = Field(description="Calculated metrics")
    recommendations: List[str] = Field(description="Actionable recommendations")
    confidence_score: float = Field(ge=0, le=1, description="Analysis confidence")
    reasoning: Optional[str] = Field(None, description="Reasoning process")


# ============================================================================
# Code Generation Models
# ============================================================================

class CodeRequest(BaseModel):
    """Request for code generation."""
    task: str = Field(description="Coding task description")
    language: str = Field(default="python", description="Programming language")
    context: Optional[str] = Field(None, description="Additional context")


class CodeOutput(BaseModel):
    """Output from code generation agent."""
    code: str = Field(description="Generated code")
    explanation: str = Field(description="Code explanation")
    tests: Optional[str] = Field(None, description="Test cases")
    dependencies: List[str] = Field(default_factory=list, description="Required dependencies")
    confidence: float = Field(ge=0, le=1, description="Generation confidence")


# ============================================================================
# Validation Models
# ============================================================================

class ValidationRequest(BaseModel):
    """Request for validation."""
    content: str = Field(description="Content to validate")
    validation_type: str = Field(description="Type of validation")
    criteria: List[str] = Field(default_factory=list, description="Validation criteria")


class ValidationOutput(BaseModel):
    """Output from validation agent."""
    is_valid: bool = Field(description="Whether content is valid")
    issues: List[str] = Field(description="Issues found")
    suggestions: List[str] = Field(description="Improvement suggestions")
    score: float = Field(ge=0, le=1, description="Quality score")


# ============================================================================
# Orchestration Models
# ============================================================================

class TaskResult(BaseModel):
    """Result from a delegated task."""
    agent: str = Field(description="Agent that executed the task")
    output: Dict[str, Any] = Field(description="Agent output")
    execution_time: float = Field(description="Execution time in seconds")
    success: bool = Field(description="Whether task succeeded")


class OrchestratorOutput(BaseModel):
    """Output from orchestrator agent."""
    task_results: Dict[str, TaskResult] = Field(description="Results from specialists")
    synthesis: str = Field(description="Combined analysis and recommendations")
    next_actions: List[str] = Field(description="Suggested next steps")
    execution_strategy: str = Field(description="Strategy used (sequential/parallel)")
    total_time: float = Field(description="Total execution time")


# ============================================================================
# Context Models
# ============================================================================

class ConversationContext(BaseModel):
    """Conversation context for maintaining state."""
    context_id: str = Field(description="Unique context identifier")
    messages: List[Dict[str, Any]] = Field(description="Message history")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: float = Field(description="Creation timestamp")
    updated_at: float = Field(description="Last update timestamp")


class Task(BaseModel):
    """Task representation."""
    task_id: str = Field(description="Unique task identifier")
    context_id: Optional[str] = Field(None, description="Associated context")
    message: str = Field(description="Task description")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Task status")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result")
    error: Optional[str] = Field(None, description="Error message if failed")
    created_at: float = Field(description="Creation timestamp")
    updated_at: float = Field(description="Last update timestamp")
    assigned_agent: Optional[str] = Field(None, description="Assigned agent")


# ============================================================================
# A2A Protocol Models
# ============================================================================

class A2ARequest(BaseModel):
    """A2A protocol request."""
    jsonrpc: str = Field(default="2.0", description="JSON-RPC version")
    id: str = Field(description="Request ID")
    method: str = Field(description="Method name")
    params: Dict[str, Any] = Field(description="Method parameters")


class A2AResponse(BaseModel):
    """A2A protocol response."""
    jsonrpc: str = Field(default="2.0", description="JSON-RPC version")
    id: str = Field(description="Request ID")
    result: Optional[Dict[str, Any]] = Field(None, description="Result data")
    error: Optional[Dict[str, Any]] = Field(None, description="Error data")


class AgentMetadata(BaseModel):
    """Agent metadata for discovery."""
    name: str = Field(description="Agent name")
    version: str = Field(description="Agent version")
    description: str = Field(description="Agent description")
    capabilities: Dict[str, Any] = Field(description="Agent capabilities")
    role: AgentRole = Field(description="Agent role")
