"""Main application entry point for A2A multi-agent system."""
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
import httpx

from config.settings import settings
from models.dependencies import (
    AnalystDependencies,
    CoderDependencies,
    ValidatorDependencies,
    OrchestratorDependencies
)
from storage.memory_storage import MemorySimpleStorage
from agents.specialists.analyst import data_analyst_agent
from agents.specialists.coder import code_generator_agent
from agents.specialists.validator import validator_agent
from agents.orchestrator import orchestrator_agent
from a2a.server import AgentMetadata, create_a2a_app


# Shared HTTP client for orchestrator
http_client: httpx.AsyncClient = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI apps."""
    global http_client
    http_client = httpx.AsyncClient(timeout=60.0)
    yield
    await http_client.aclose()


# ============================================================================
# Analyst Agent Setup
# ============================================================================

def create_analyst_deps(**kwargs):
    """Factory for analyst dependencies."""
    return AnalystDependencies(
        agent_id="data-analyst",
        storage=MemorySimpleStorage(),
        cache_enabled=True
    )


analyst_metadata = AgentMetadata(
    name="Data Analyst Agent",
    version="1.0.0",
    description="Specialist agent for data analysis, statistics, and trend identification",
    capabilities={
        "tools": [
            "calculate_statistics",
            "analyze_trends",
            "compare_datasets",
            "cache_result"
        ],
        "model": settings.analyst_model,
        "strengths": ["data analysis", "statistical reasoning", "pattern recognition"]
    }
)

analyst_app = create_a2a_app(
    agent=data_analyst_agent,
    metadata=analyst_metadata,
    deps_factory=create_analyst_deps
)
analyst_app.router.lifespan_context = lifespan


# ============================================================================
# Coder Agent Setup
# ============================================================================

def create_coder_deps(**kwargs):
    """Factory for coder dependencies."""
    return CoderDependencies(
        agent_id="code-generator",
        storage=MemorySimpleStorage(),
        language="python",
        include_tests=True
    )


coder_metadata = AgentMetadata(
    name="Code Generator Agent",
    version="1.0.0",
    description="Specialist agent for code generation, testing, and validation",
    capabilities={
        "tools": [
            "validate_syntax",
            "generate_tests",
            "find_dependencies",
            "format_code",
            "get_code_template"
        ],
        "model": settings.coder_model,
        "languages": ["python", "javascript", "typescript", "go"],
        "strengths": ["code generation", "syntax validation", "test generation"]
    }
)

coder_app = create_a2a_app(
    agent=code_generator_agent,
    metadata=coder_metadata,
    deps_factory=create_coder_deps
)
coder_app.router.lifespan_context = lifespan


# ============================================================================
# Validator Agent Setup
# ============================================================================

def create_validator_deps(**kwargs):
    """Factory for validator dependencies."""
    return ValidatorDependencies(
        agent_id="validator",
        storage=MemorySimpleStorage(),
        strict_mode=False
    )


validator_metadata = AgentMetadata(
    name="Validator Agent",
    version="1.0.0",
    description="Specialist agent for validation, quality assurance, and format checking",
    capabilities={
        "tools": [
            "check_format",
            "check_length",
            "check_patterns",
            "check_completeness",
            "validate_against_schema"
        ],
        "model": settings.fast_model,
        "formats": ["json", "xml", "email", "url", "phone"],
        "strengths": ["fast validation", "format checking", "quality assurance"]
    }
)

validator_app = create_a2a_app(
    agent=validator_agent,
    metadata=validator_metadata,
    deps_factory=create_validator_deps
)
validator_app.router.lifespan_context = lifespan


# ============================================================================
# Orchestrator Agent Setup
# ============================================================================

def create_orchestrator_deps(**kwargs):
    """Factory for orchestrator dependencies."""
    return OrchestratorDependencies(
        agent_id="orchestrator",
        specialist_agents={
            "analyst": f"http://{settings.host}:{settings.analyst_port}",
            "coder": f"http://{settings.host}:{settings.coder_port}",
            "validator": f"http://{settings.host}:{settings.validator_port}",
        },
        http_client=http_client,
        task_storage=MemorySimpleStorage(),
        context_storage=None
    )


orchestrator_metadata = AgentMetadata(
    name="Orchestrator Agent",
    version="1.0.0",
    description="Central coordinator for multi-agent task delegation and synthesis",
    capabilities={
        "tools": [
            "delegate_to_specialist",
            "delegate_parallel",
            "get_specialist_capabilities",
            "analyze_task_complexity"
        ],
        "model": settings.orchestrator_model,
        "specialists": ["analyst", "coder", "validator"],
        "execution_modes": ["sequential", "parallel", "hybrid"],
        "strengths": ["task coordination", "result synthesis", "parallel execution"]
    }
)

orchestrator_app = create_a2a_app(
    agent=orchestrator_agent,
    metadata=orchestrator_metadata,
    deps_factory=create_orchestrator_deps
)
orchestrator_app.router.lifespan_context = lifespan


# ============================================================================
# Export apps for uvicorn
# ============================================================================

# These can be run individually:
# uvicorn main:analyst_app --port 8001
# uvicorn main:coder_app --port 8002
# uvicorn main:validator_app --port 8003
# uvicorn main:orchestrator_app --port 8000

__all__ = [
    "analyst_app",
    "coder_app",
    "validator_app",
    "orchestrator_app"
]


# ============================================================================
# Development server runner
# ============================================================================

async def run_all_servers():
    """
    Run all agent servers concurrently (for development).

    Note: In production, use docker-compose or separate processes.
    """
    import uvicorn

    # Create server configs
    servers = [
        uvicorn.Config(
            "main:orchestrator_app",
            host=settings.host,
            port=settings.orchestrator_port,
            log_level=settings.log_level.lower()
        ),
        uvicorn.Config(
            "main:analyst_app",
            host=settings.host,
            port=settings.analyst_port,
            log_level=settings.log_level.lower()
        ),
        uvicorn.Config(
            "main:coder_app",
            host=settings.host,
            port=settings.coder_port,
            log_level=settings.log_level.lower()
        ),
        uvicorn.Config(
            "main:validator_app",
            host=settings.host,
            port=settings.validator_port,
            log_level=settings.log_level.lower()
        ),
    ]

    # Run servers concurrently
    server_tasks = [uvicorn.Server(config).serve() for config in servers]
    await asyncio.gather(*server_tasks)


if __name__ == "__main__":
    print("🚀 Starting A2A Multi-Agent Orchestration System")
    print(f"📊 Orchestrator: http://{settings.host}:{settings.orchestrator_port}")
    print(f"📈 Analyst: http://{settings.host}:{settings.analyst_port}")
    print(f"💻 Coder: http://{settings.host}:{settings.coder_port}")
    print(f"✅ Validator: http://{settings.host}:{settings.validator_port}")
    print("\nNote: Run with separate uvicorn commands or docker-compose for production")
    print("Example: uvicorn main:orchestrator_app --port 8000\n")

    # For development, you could run:
    # asyncio.run(run_all_servers())
