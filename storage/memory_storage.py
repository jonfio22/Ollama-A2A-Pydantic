"""In-memory storage implementations for testing."""
import time
from typing import Optional, List, Dict, Any
from models.schemas import Task, ConversationContext, TaskStatus
from storage.interfaces import TaskStorage, ContextStorage, SimpleStorage


class MemoryTaskStorage(TaskStorage):
    """In-memory task storage for testing."""

    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.context_index: Dict[str, set] = {}
        self.status_index: Dict[TaskStatus, set] = {status: set() for status in TaskStatus}

    async def create_task(self, task: Task) -> Task:
        """Store task in memory."""
        self.tasks[task.task_id] = task

        # Index by context
        if task.context_id:
            if task.context_id not in self.context_index:
                self.context_index[task.context_id] = set()
            self.context_index[task.context_id].add(task.task_id)

        # Index by status
        self.status_index[task.status].add(task.task_id)

        return task

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieve task from memory."""
        return self.tasks.get(task_id)

    async def update_task(self, task_id: str, updates: dict) -> Task:
        """Update task fields."""
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")

        old_status = task.status

        # Apply updates
        for key, value in updates.items():
            setattr(task, key, value)

        task.updated_at = time.time()

        # Update status index if changed
        if old_status != task.status:
            self.status_index[old_status].discard(task_id)
            self.status_index[task.status].add(task_id)

        return task

    async def list_tasks(
        self,
        context_id: Optional[str] = None,
        status: Optional[TaskStatus] = None,
        limit: int = 100
    ) -> List[Task]:
        """List tasks with filters."""
        if context_id:
            task_ids = self.context_index.get(context_id, set())
        elif status:
            task_ids = self.status_index[status]
        else:
            task_ids = set(self.tasks.keys())

        tasks = [self.tasks[tid] for tid in task_ids if tid in self.tasks]
        return tasks[:limit]

    async def delete_task(self, task_id: str) -> bool:
        """Delete task."""
        task = self.tasks.pop(task_id, None)
        if not task:
            return False

        # Remove from indexes
        if task.context_id:
            self.context_index.get(task.context_id, set()).discard(task_id)
        self.status_index[task.status].discard(task_id)

        return True


class MemoryContextStorage(ContextStorage):
    """In-memory context storage for testing."""

    def __init__(self):
        self.contexts: Dict[str, ConversationContext] = {}

    async def create_context(self, context: ConversationContext) -> ConversationContext:
        """Create new context."""
        self.contexts[context.context_id] = context
        return context

    async def get_context(self, context_id: str) -> Optional[ConversationContext]:
        """Retrieve context."""
        return self.contexts.get(context_id)

    async def append_message(
        self,
        context_id: str,
        message: Dict[str, Any]
    ) -> ConversationContext:
        """Append message to context."""
        context = self.contexts.get(context_id)
        if not context:
            raise ValueError(f"Context not found: {context_id}")

        context.messages.append(message)
        context.updated_at = time.time()

        return context

    async def update_context(
        self,
        context_id: str,
        updates: dict
    ) -> ConversationContext:
        """Update context metadata."""
        context = self.contexts.get(context_id)
        if not context:
            raise ValueError(f"Context not found: {context_id}")

        for key, value in updates.items():
            if key != 'messages':
                setattr(context, key, value)

        context.updated_at = time.time()

        return context

    async def delete_context(self, context_id: str) -> bool:
        """Delete context."""
        return self.contexts.pop(context_id, None) is not None


class MemorySimpleStorage(SimpleStorage):
    """In-memory simple key-value storage."""

    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.expiry: Dict[str, float] = {}

    async def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        # Check expiry
        if key in self.expiry:
            if time.time() > self.expiry[key]:
                await self.delete(key)
                return None

        return self.data.get(key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value with optional TTL."""
        self.data[key] = value
        if ttl:
            self.expiry[key] = time.time() + ttl

    async def delete(self, key: str) -> bool:
        """Delete key."""
        self.data.pop(key, None)
        self.expiry.pop(key, None)
        return True

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        value = await self.get(key)
        return value is not None
