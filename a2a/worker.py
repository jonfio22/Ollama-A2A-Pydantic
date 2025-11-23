"""Custom A2A worker implementation for Pydantic AI agents."""
from typing import Any, Dict, Optional
from pydantic_ai import Agent


class PydanticAIWorker:
    """
    Custom worker that wraps Pydantic AI agents for A2A protocol compliance.

    This worker handles the translation between A2A protocol messages
    and Pydantic AI agent inputs/outputs.
    """

    def __init__(self, agent: Agent, deps_factory=None):
        """
        Initialize the worker.

        Args:
            agent: Pydantic AI Agent instance
            deps_factory: Optional factory function to create agent dependencies
        """
        self.agent = agent
        self.deps_factory = deps_factory

    async def run(
        self,
        message: str,
        context_id: Optional[str] = None,
        artifacts: Optional[list[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Execute agent with A2A protocol compliance.

        Args:
            message: User message/query
            context_id: Optional context identifier for maintaining conversation state
            artifacts: Optional list of artifacts (additional context data)

        Returns:
            A2A-compliant response dictionary
        """
        # Prepare dependencies if factory is provided
        deps = None
        if self.deps_factory:
            deps = self.deps_factory(context_id=context_id, artifacts=artifacts)

        # Run the agent
        try:
            result = await self.agent.run(message, deps=deps)

            # Extract output
            if hasattr(result.data, 'model_dump'):
                output = result.data.model_dump()
            else:
                output = str(result.data)

            # Build A2A response
            response = {
                "output": output,
                "metadata": {
                    "model": str(self.agent.model),
                    "messages_count": len(result.all_messages()),
                    "context_id": context_id
                }
            }

            # Add cost information if available
            try:
                cost_info = result.cost()
                if cost_info:
                    response["metadata"]["cost"] = {
                        "total_tokens": getattr(cost_info, 'total_tokens', 0),
                        "request_tokens": getattr(cost_info, 'request_tokens', 0),
                        "response_tokens": getattr(cost_info, 'response_tokens', 0),
                    }
            except:
                pass

            return response

        except Exception as e:
            return {
                "error": {
                    "message": str(e),
                    "type": type(e).__name__
                },
                "metadata": {
                    "context_id": context_id
                }
            }

    async def stream(
        self,
        message: str,
        context_id: Optional[str] = None,
        artifacts: Optional[list[Dict[str, Any]]] = None
    ):
        """
        Stream agent responses (for future implementation).

        Args:
            message: User message/query
            context_id: Optional context identifier
            artifacts: Optional artifacts

        Yields:
            Streamed response chunks
        """
        # Prepare dependencies
        deps = None
        if self.deps_factory:
            deps = self.deps_factory(context_id=context_id, artifacts=artifacts)

        # Stream responses
        async with self.agent.run_stream(message, deps=deps) as result:
            async for chunk in result.stream():
                yield {
                    "chunk": str(chunk),
                    "context_id": context_id
                }
