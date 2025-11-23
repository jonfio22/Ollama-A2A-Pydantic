"""Application configuration management."""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    # Ollama configuration
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Redis configuration
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")

    # Application settings
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    service_name: str = os.getenv("SERVICE_NAME", "a2a-orchestration")

    # Agent model configuration
    orchestrator_model: str = os.getenv("ORCHESTRATOR_MODEL", "ollama:llama3.1:8b")
    analyst_model: str = os.getenv("ANALYST_MODEL", "ollama:qwen2.5:7b")
    coder_model: str = os.getenv("CODER_MODEL", "ollama:deepseek-coder-v2:16b")
    fast_model: str = os.getenv("FAST_MODEL", "ollama:llama3.2:3b")

    # Server configuration
    host: str = os.getenv("HOST", "0.0.0.0")
    orchestrator_port: int = int(os.getenv("ORCHESTRATOR_PORT", "8000"))
    analyst_port: int = int(os.getenv("ANALYST_PORT", "8001"))
    coder_port: int = int(os.getenv("CODER_PORT", "8002"))
    validator_port: int = int(os.getenv("VALIDATOR_PORT", "8003"))


# Global settings instance
settings = Settings()
