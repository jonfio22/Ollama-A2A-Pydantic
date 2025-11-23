"""A2A client for communicating with A2A-compliant agents."""
import httpx
from typing import Optional, Dict, Any
from uuid import uuid4


class A2AClient:
    """
    Client for communicating with A2A-compliant agents.

    Implements the A2A protocol for sending messages and receiving responses
    from agent endpoints.
    """

    def __init__(self, base_url: str, timeout: float = 60.0):
        """
        Initialize A2A client.

        Args:
            base_url: Base URL of the A2A agent endpoint
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.client:
            await self.client.aclose()

    async def send_message(
        self,
        message: str,
        context_id: Optional[str] = None,
        artifacts: Optional[list] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send message to A2A agent.

        Args:
            message: Message to send
            context_id: Optional context identifier
            artifacts: Optional list of artifacts
            request_id: Optional custom request ID

        Returns:
            Agent response dictionary

        Raises:
            httpx.HTTPError: If request fails
        """
        if not self.client:
            self.client = httpx.AsyncClient(timeout=self.timeout)

        payload = {
            "jsonrpc": "2.0",
            "id": request_id or str(uuid4()),
            "method": "run",
            "params": {
                "message": message,
            }
        }

        if context_id:
            payload["params"]["context_id"] = context_id
        if artifacts:
            payload["params"]["artifacts"] = artifacts

        response = await self.client.post(
            f"{self.base_url}/a2a/run",
            json=payload
        )
        response.raise_for_status()
        return response.json()

    async def get_agent_metadata(self) -> Dict[str, Any]:
        """
        Retrieve agent capabilities and metadata.

        Returns:
            Agent metadata dictionary

        Raises:
            httpx.HTTPError: If request fails
        """
        if not self.client:
            self.client = httpx.AsyncClient(timeout=self.timeout)

        response = await self.client.get(
            f"{self.base_url}/.well-known/agent.json"
        )
        response.raise_for_status()
        return response.json()

    async def close(self):
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose()
            self.client = None


async def send_to_agent(
    endpoint: str,
    message: str,
    context_id: Optional[str] = None,
    artifacts: Optional[list] = None,
    timeout: float = 60.0
) -> Dict[str, Any]:
    """
    Convenience function to send a message to an agent.

    Args:
        endpoint: Agent endpoint URL
        message: Message to send
        context_id: Optional context identifier
        artifacts: Optional artifacts
        timeout: Request timeout

    Returns:
        Agent response
    """
    async with A2AClient(endpoint, timeout=timeout) as client:
        return await client.send_message(
            message=message,
            context_id=context_id,
            artifacts=artifacts
        )
