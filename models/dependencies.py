"""Dependency injection models for agents."""
from dataclasses import dataclass
from typing import Dict, Protocol, List, Any, Optional
import httpx


# ============================================================================
# Protocol Definitions (Interfaces)
# ============================================================================

class StorageInterface(Protocol):
    """Interface for storage backends."""

    async def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        ...

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value with optional TTL."""
        ...

    async def delete(self, key: str) -> None:
        """Delete key."""
        ...


class TaskStorageInterface(Protocol):
    """Interface for task storage."""

    async def create_task(self, task: Any) -> Any:
        """Create a new task."""
        ...

    async def get_task(self, task_id: str) -> Optional[Any]:
        """Get task by ID."""
        ...

    async def update_task(self, task_id: str, updates: dict) -> Any:
        """Update task."""
        ...

    async def list_tasks(self, **filters) -> List[Any]:
        """List tasks with filters."""
        ...


class ContextStorageInterface(Protocol):
    """Interface for context storage."""

    async def create_context(self, context: Any) -> Any:
        """Create new context."""
        ...

    async def get_context(self, context_id: str) -> Optional[Any]:
        """Get context by ID."""
        ...

    async def append_message(self, context_id: str, message: Dict[str, Any]) -> Any:
        """Append message to context."""
        ...


# ============================================================================
# Dependency Classes
# ============================================================================

@dataclass
class BaseAgentDependencies:
    """Base dependencies for all agents."""
    agent_id: str
    storage: StorageInterface


@dataclass
class OrchestratorDependencies:
    """Dependencies for orchestrator agent."""
    agent_id: str
    specialist_agents: Dict[str, str]  # agent_name -> endpoint_url
    http_client: httpx.AsyncClient
    task_storage: Optional[TaskStorageInterface] = None
    context_storage: Optional[ContextStorageInterface] = None


@dataclass
class AnalystDependencies:
    """Dependencies for data analyst agent."""
    agent_id: str
    storage: StorageInterface
    cache_enabled: bool = True


@dataclass
class CoderDependencies:
    """Dependencies for code generation agent."""
    agent_id: str
    storage: StorageInterface
    language: str = "python"
    include_tests: bool = True


@dataclass
class ValidatorDependencies:
    """Dependencies for validation agent."""
    agent_id: str
    storage: StorageInterface
    strict_mode: bool = False
