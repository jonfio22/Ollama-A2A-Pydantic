"""Abstract storage interfaces."""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from models.schemas import Task, ConversationContext, TaskStatus


class TaskStorage(ABC):
    """Abstract interface for task persistence."""

    @abstractmethod
    async def create_task(self, task: Task) -> Task:
        """Create new task."""
        pass

    @abstractmethod
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieve task by ID."""
        pass

    @abstractmethod
    async def update_task(self, task_id: str, updates: dict) -> Task:
        """Update task fields."""
        pass

    @abstractmethod
    async def list_tasks(
        self,
        context_id: Optional[str] = None,
        status: Optional[TaskStatus] = None,
        limit: int = 100
    ) -> List[Task]:
        """List tasks with optional filters."""
        pass

    @abstractmethod
    async def delete_task(self, task_id: str) -> bool:
        """Delete task."""
        pass


class ContextStorage(ABC):
    """Abstract interface for conversation context."""

    @abstractmethod
    async def create_context(self, context: ConversationContext) -> ConversationContext:
        """Create new context."""
        pass

    @abstractmethod
    async def get_context(self, context_id: str) -> Optional[ConversationContext]:
        """Retrieve context by ID."""
        pass

    @abstractmethod
    async def append_message(
        self,
        context_id: str,
        message: Dict[str, Any]
    ) -> ConversationContext:
        """Append message to context."""
        pass

    @abstractmethod
    async def update_context(
        self,
        context_id: str,
        updates: dict
    ) -> ConversationContext:
        """Update context metadata."""
        pass

    @abstractmethod
    async def delete_context(self, context_id: str) -> bool:
        """Delete context."""
        pass


class SimpleStorage(ABC):
    """Simple key-value storage interface."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value with optional TTL in seconds."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass
