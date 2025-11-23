"""A2A server setup and FastAPI integration."""
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydantic_ai import Agent
from a2a.worker import PydanticAIWorker


class A2ARunRequest(BaseModel):
    """A2A run request model."""
    jsonrpc: str = "2.0"
    id: str
    method: str
    params: Dict[str, Any]


class A2ARunResponse(BaseModel):
    """A2A run response model."""
    jsonrpc: str = "2.0"
    id: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


class AgentMetadata(BaseModel):
    """Agent metadata model."""
    name: str
    version: str
    description: str
    capabilities: Dict[str, Any]


def create_a2a_app(
    agent: Agent,
    metadata: AgentMetadata,
    deps_factory=None,
    title: Optional[str] = None,
    description: Optional[str] = None
) -> FastAPI:
    """
    Create a FastAPI app with A2A protocol endpoints.

    Args:
        agent: Pydantic AI Agent instance
        metadata: Agent metadata
        deps_factory: Optional factory function for creating agent dependencies
        title: Optional app title (defaults to metadata.name)
        description: Optional app description (defaults to metadata.description)

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title=title or metadata.name,
        description=description or metadata.description,
        version=metadata.version
    )

    # Create worker
    worker = PydanticAIWorker(agent=agent, deps_factory=deps_factory)

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "name": metadata.name,
            "version": metadata.version,
            "endpoints": {
                "run": "/a2a/run",
                "metadata": "/.well-known/agent.json",
                "health": "/health"
            }
        }

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.post("/a2a/run", response_model=A2ARunResponse)
    async def run_agent(request: A2ARunRequest) -> A2ARunResponse:
        """
        A2A protocol run endpoint.

        Args:
            request: A2A run request

        Returns:
            A2A run response
        """
        if request.method != "run":
            return A2ARunResponse(
                id=request.id,
                error={
                    "code": -32601,
                    "message": f"Method not found: {request.method}"
                }
            )

        try:
            # Extract parameters
            message = request.params.get("message")
            if not message:
                raise ValueError("Missing required parameter: message")

            context_id = request.params.get("context_id")
            artifacts = request.params.get("artifacts")

            # Run agent
            result = await worker.run(
                message=message,
                context_id=context_id,
                artifacts=artifacts
            )

            # Check for errors in result
            if "error" in result:
                return A2ARunResponse(
                    id=request.id,
                    error={
                        "code": -32000,
                        "message": result["error"].get("message", "Unknown error"),
                        "data": result.get("metadata")
                    }
                )

            return A2ARunResponse(
                id=request.id,
                result=result
            )

        except Exception as e:
            return A2ARunResponse(
                id=request.id,
                error={
                    "code": -32000,
                    "message": str(e),
                    "data": {"type": type(e).__name__}
                }
            )

    @app.get("/.well-known/agent.json")
    async def agent_metadata():
        """
        Agent metadata discovery endpoint.

        Returns:
            Agent metadata
        """
        return metadata.model_dump()

    return app


def quick_serve_agent(
    agent: Agent,
    name: str,
    version: str = "1.0.0",
    description: str = "A2A Agent",
    capabilities: Optional[Dict[str, Any]] = None,
    deps_factory=None
) -> FastAPI:
    """
    Quick setup function to serve an agent with A2A protocol.

    Args:
        agent: Pydantic AI Agent
        name: Agent name
        version: Agent version
        description: Agent description
        capabilities: Agent capabilities dictionary
        deps_factory: Optional dependencies factory

    Returns:
        Configured FastAPI app
    """
    metadata = AgentMetadata(
        name=name,
        version=version,
        description=description,
        capabilities=capabilities or {}
    )

    return create_a2a_app(
        agent=agent,
        metadata=metadata,
        deps_factory=deps_factory
    )
