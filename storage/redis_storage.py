"""Redis-based storage implementations."""
import json
import time
from typing import Optional, List, Dict, Any
from redis.asyncio import Redis, from_url
from models.schemas import Task, ConversationContext, TaskStatus
from storage.interfaces import TaskStorage, ContextStorage, SimpleStorage


class RedisTaskStorage(TaskStorage):
    """Redis-based task storage."""

    def __init__(self, redis_url: str):
        self.redis: Optional[Redis] = None
        self.redis_url = redis_url

    async def connect(self):
        """Connect to Redis."""
        if not self.redis:
            self.redis = await from_url(self.redis_url, decode_responses=True)

    async def close(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()

    async def create_task(self, task: Task) -> Task:
        """Store task in Redis."""
        await self.connect()
        key = f"task:{task.task_id}"
        await self.redis.set(
            key,
            task.model_dump_json(),
            ex=86400  # 24-hour TTL
        )

        # Index by context_id
        if task.context_id:
            await self.redis.sadd(
                f"context:{task.context_id}:tasks",
                task.task_id
            )

        # Index by status
        await self.redis.sadd(
            f"tasks:status:{task.status.value}",
            task.task_id
        )

        return task

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieve task from Redis."""
        await self.connect()
        data = await self.redis.get(f"task:{task_id}")
        if not data:
            return None
        return Task.model_validate_json(data)

    async def update_task(self, task_id: str, updates: dict) -> Task:
        """Update task fields atomically."""
        task = await self.get_task(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")

        # Remove from old status index
        old_status = task.status

        # Apply updates
        for key, value in updates.items():
            setattr(task, key, value)

        task.updated_at = time.time()

        # Save back to Redis
        await self.redis.set(
            f"task:{task_id}",
            task.model_dump_json(),
            ex=86400
        )

        # Update status index if changed
        if old_status != task.status:
            await self.redis.srem(f"tasks:status:{old_status.value}", task_id)
            await self.redis.sadd(f"tasks:status:{task.status.value}", task_id)

        return task

    async def list_tasks(
        self,
        context_id: Optional[str] = None,
        status: Optional[TaskStatus] = None,
        limit: int = 100
    ) -> List[Task]:
        """List tasks with filters."""
        await self.connect()

        # Determine which index to use
        if context_id:
            task_ids = await self.redis.smembers(f"context:{context_id}:tasks")
        elif status:
            task_ids = await self.redis.smembers(f"tasks:status:{status.value}")
        else:
            # Get all tasks from all status indexes
            all_ids = set()
            for s in TaskStatus:
                ids = await self.redis.smembers(f"tasks:status:{s.value}")
                all_ids.update(ids)
            task_ids = all_ids

        # Fetch tasks
        tasks = []
        for task_id in list(task_ids)[:limit]:
            task = await self.get_task(task_id)
            if task:
                tasks.append(task)

        return tasks

    async def delete_task(self, task_id: str) -> bool:
        """Delete task."""
        await self.connect()
        task = await self.get_task(task_id)
        if not task:
            return False

        # Remove from indexes
        if task.context_id:
            await self.redis.srem(f"context:{task.context_id}:tasks", task_id)
        await self.redis.srem(f"tasks:status:{task.status.value}", task_id)

        # Delete task
        await self.redis.delete(f"task:{task_id}")
        return True


class RedisContextStorage(ContextStorage):
    """Redis-based context storage."""

    def __init__(self, redis_url: str):
        self.redis: Optional[Redis] = None
        self.redis_url = redis_url

    async def connect(self):
        """Connect to Redis."""
        if not self.redis:
            self.redis = await from_url(self.redis_url, decode_responses=True)

    async def close(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()

    async def create_context(self, context: ConversationContext) -> ConversationContext:
        """Create new context."""
        await self.connect()
        key = f"context:{context.context_id}"
        await self.redis.set(
            key,
            context.model_dump_json(),
            ex=86400  # 24-hour TTL
        )
        return context

    async def get_context(self, context_id: str) -> Optional[ConversationContext]:
        """Retrieve context."""
        await self.connect()
        data = await self.redis.get(f"context:{context_id}")
        if not data:
            return None
        return ConversationContext.model_validate_json(data)

    async def append_message(
        self,
        context_id: str,
        message: Dict[str, Any]
    ) -> ConversationContext:
        """Append message to context."""
        context = await self.get_context(context_id)
        if not context:
            raise ValueError(f"Context not found: {context_id}")

        context.messages.append(message)
        context.updated_at = time.time()

        await self.redis.set(
            f"context:{context_id}",
            context.model_dump_json(),
            ex=86400
        )

        return context

    async def update_context(
        self,
        context_id: str,
        updates: dict
    ) -> ConversationContext:
        """Update context metadata."""
        context = await self.get_context(context_id)
        if not context:
            raise ValueError(f"Context not found: {context_id}")

        for key, value in updates.items():
            if key != 'messages':  # Don't allow direct message updates
                setattr(context, key, value)

        context.updated_at = time.time()

        await self.redis.set(
            f"context:{context_id}",
            context.model_dump_json(),
            ex=86400
        )

        return context

    async def delete_context(self, context_id: str) -> bool:
        """Delete context."""
        await self.connect()
        result = await self.redis.delete(f"context:{context_id}")
        return result > 0


class RedisSimpleStorage(SimpleStorage):
    """Redis-based simple key-value storage."""

    def __init__(self, redis_url: str):
        self.redis: Optional[Redis] = None
        self.redis_url = redis_url

    async def connect(self):
        """Connect to Redis."""
        if not self.redis:
            self.redis = await from_url(self.redis_url, decode_responses=True)

    async def close(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()

    async def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        await self.connect()
        data = await self.redis.get(key)
        if data is None:
            return None
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return data

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value with optional TTL."""
        await self.connect()
        # Serialize if not string
        if not isinstance(value, str):
            value = json.dumps(value)

        if ttl:
            await self.redis.set(key, value, ex=ttl)
        else:
            await self.redis.set(key, value)

    async def delete(self, key: str) -> bool:
        """Delete key."""
        await self.connect()
        result = await self.redis.delete(key)
        return result > 0

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        await self.connect()
        return await self.redis.exists(key) > 0
