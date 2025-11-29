# Architecture Documentation

## System Overview

The A2A Multi-Agent Orchestration System is built on a layered architecture that enables sophisticated multi-agent workflows with type safety, observability, and scalability.

## Core Components

### 1. Agent Layer

#### Base Agent Framework (`agents/base.py`)

Factory functions for creating standardized agents:
- `create_agent()` - General purpose agent factory
- `create_fast_agent()` - Fast agents using llama3.2:3b
- `create_analytical_agent()` - Analytical agents using qwen2.5:7b
- `create_coding_agent()` - Coding agents using deepseek-coder-v2:16b
- `create_orchestrator_agent()` - Orchestrators using llama3.1:8b

#### Specialist Agents

**Data Analyst** (`agents/specialists/analyst.py`)
- Model: qwen2.5:7b
- Tools: Statistics, trend analysis, dataset comparison
- Use case: Data analysis and pattern recognition

**Code Generator** (`agents/specialists/coder.py`)
- Model: deepseek-coder-v2:16b
- Tools: Syntax validation, test generation, dependency extraction
- Use case: Code generation and validation

**Validator** (`agents/specialists/validator.py`)
- Model: llama3.2:3b
- Tools: Format checking, pattern matching, schema validation
- Use case: Fast quality assurance

#### Orchestrator Agent (`agents/orchestrator.py`)

- Model: llama3.1:8b
- Capabilities:
  - Task complexity analysis
  - Sequential/parallel delegation
  - Result synthesis
  - Intermediate result caching

### 2. A2A Protocol Layer

#### Worker (`a2a/worker.py`)

Wraps Pydantic AI agents for A2A protocol compliance:
- Translates A2A messages to agent inputs
- Converts agent outputs to A2A responses
- Handles context and artifacts
- Manages errors and metadata

#### Client (`a2a/client.py`)

A2A client for making requests:
- JSON-RPC message formatting
- Context management
- Artifact handling
- Metadata discovery

#### Server (`a2a/server.py`)

FastAPI server setup:
- A2A protocol endpoints (`/a2a/run`)
- Agent metadata endpoint (`/.well-known/agent.json`)
- Health checks
- Error handling

### 3. Storage Layer

#### Interfaces (`storage/interfaces.py`)

Abstract base classes:
- `TaskStorage` - Task persistence
- `ContextStorage` - Conversation context
- `SimpleStorage` - Key-value storage

#### Implementations

**Redis Storage** (`storage/redis_storage.py`)
- Production-ready persistence
- TTL support
- Index management
- Atomic operations

**Memory Storage** (`storage/memory_storage.py`)
- In-memory implementation
- Testing support
- No external dependencies

### 4. Model Layer

#### Schemas (`models/schemas.py`)

Pydantic models for all I/O:
- Base models (BaseAgentOutput)
- Specialist outputs (AnalysisOutput, CodeOutput, ValidationOutput)
- Orchestration models (OrchestratorOutput, TaskResult)
- A2A protocol models (A2ARequest, A2AResponse)
- Storage models (Task, ConversationContext)

#### Dependencies (`models/dependencies.py`)

Dependency injection:
- Protocol definitions (StorageInterface, etc.)
- Agent-specific dependencies
- Type-safe dependency containers

### 5. Configuration Layer

#### Settings (`config/settings.py`)

Environment-based configuration:
- Ollama settings
- Redis connection
- Agent model selection
- Server ports

## Data Flow

### 1. Simple Request Flow

```
User Request
    │
    ▼
A2A Client
    │
    ├─ Create JSON-RPC payload
    ├─ Add context_id (optional)
    └─ Add artifacts (optional)
    │
    ▼
HTTP POST /a2a/run
    │
    ▼
FastAPI Server
    │
    ├─ Validate request
    ├─ Extract parameters
    └─ Route to worker
    │
    ▼
PydanticAIWorker
    │
    ├─ Create dependencies
    ├─ Run agent
    └─ Format response
    │
    ▼
Pydantic AI Agent
    │
    ├─ Process message
    ├─ Call tools (if needed)
    ├─ Generate response
    └─ Return structured output
    │
    ▼
A2A Response
    │
    └─ JSON-RPC result
```

### 2. Orchestrated Request Flow

```
User → Orchestrator
         │
         ├─ Analyze task complexity
         │
         ├─ Determine strategy
         │   ├─ Sequential
         │   ├─ Parallel
         │   └─ Hybrid
         │
         ├─ Delegate to specialists
         │   │
         │   ├─ Analyst (A2A)
         │   │   └─ Calculate statistics
         │   │
         │   ├─ Coder (A2A)
         │   │   └─ Generate code
         │   │
         │   └─ Validator (A2A)
         │       └─ Validate output
         │
         ├─ Aggregate results
         │
         ├─ Synthesize findings
         │
         └─ Return comprehensive output
```

### 3. Parallel Execution Flow

```
Orchestrator receives task
    │
    ├─ Task 1 → Analyst ─┐
    ├─ Task 2 → Coder ───┼─ asyncio.gather()
    └─ Task 3 → Validator┘
                │
                ▼
        Wait for all completions
                │
                ▼
        Aggregate results
                │
                ▼
        Synthesize output
```

## Communication Patterns

### 1. A2A Protocol

Standard JSON-RPC over HTTP:

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": "unique-id",
  "method": "run",
  "params": {
    "message": "User query",
    "context_id": "conversation-id",
    "artifacts": [{"key": "value"}]
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": "unique-id",
  "result": {
    "output": {...},
    "metadata": {...}
  }
}
```

### 2. Tool Calling

Pydantic AI tool pattern:

```python
@agent.tool
async def tool_name(
    ctx: RunContext[DepsType],
    param: str
) -> ReturnType:
    """Tool description for LLM."""
    # Access dependencies
    storage = ctx.deps.storage

    # Tool logic
    result = do_work(param)

    return result
```

### 3. Dependency Injection

Type-safe dependencies:

```python
@dataclass
class AgentDeps:
    agent_id: str
    storage: StorageInterface
    http_client: httpx.AsyncClient

# Factory pattern
def create_deps(**kwargs) -> AgentDeps:
    return AgentDeps(
        agent_id="agent-id",
        storage=MemorySimpleStorage(),
        http_client=httpx.AsyncClient()
    )

# Agent uses deps
result = await agent.run(message, deps=create_deps())
```

## Scalability Considerations

### Horizontal Scaling

1. **Stateless agents** - Each agent can run in multiple instances
2. **Load balancing** - Distribute requests across instances
3. **Redis storage** - Shared state across instances

### Vertical Scaling

1. **Model selection** - Choose appropriate model sizes
2. **GPU acceleration** - Enable CUDA for faster inference
3. **Model caching** - Keep models in memory

### Performance Optimization

1. **Parallel execution** - Independent tasks run concurrently
2. **Connection pooling** - Reuse HTTP connections
3. **Result caching** - Cache expensive computations
4. **Model quantization** - Use smaller quantized models

## Security

### Authentication

- API key validation (optional)
- Rate limiting
- Request validation

### Data Privacy

- All processing local (Ollama)
- No external API calls
- Full data control

### Input Validation

- Pydantic models validate all inputs
- Injection attack prevention
- Length limits
- Format validation

## Monitoring & Observability

### Health Checks

- `/health` endpoint on each agent
- Docker healthchecks
- K8s liveness/readiness probes

### Metrics

- Request count
- Response times
- Error rates
- Model performance

### Logging

- Structured logging
- Request/response logging
- Error tracking
- Tool call tracing

## Deployment Patterns

### Development

```bash
# Individual processes
uvicorn main:orchestrator_app --port 8000
uvicorn main:analyst_app --port 8001
uvicorn main:coder_app --port 8002
uvicorn main:validator_app --port 8003
```

### Production (Docker)

```yaml
# docker-compose.yml
services:
  ollama:    # Shared LLM runtime
  redis:     # Shared storage
  orchestrator:
  analyst:
  coder:
  validator:
```

### Production (Kubernetes)

```yaml
# Deployments for each agent
# Services for load balancing
# ConfigMaps for configuration
# Secrets for sensitive data
```

## Extension Points

### Adding New Agents

1. Define output schema
2. Create dependencies
3. Implement agent with tools
4. Register in main.py
5. Add A2A endpoint

### Adding New Tools

1. Define tool function
2. Add `@agent.tool` decorator
3. Document with docstring
4. Implement tool logic

### Custom Storage

1. Implement storage interfaces
2. Add configuration
3. Update dependency factories

### Custom Protocols

1. Create custom worker
2. Implement protocol endpoints
3. Update server setup

## Best Practices

1. **Type Safety** - Use Pydantic models everywhere
2. **Dependency Injection** - Never hardcode dependencies
3. **Error Handling** - Always handle errors gracefully
4. **Testing** - Write tests for all agents and tools
5. **Documentation** - Document all tools and APIs
6. **Monitoring** - Implement observability from day one
7. **Security** - Validate all inputs, implement rate limiting
8. **Performance** - Use appropriate models, enable parallelization

## Future Enhancements

1. **Streaming responses** - Real-time output streaming
2. **Agent registry** - Dynamic agent discovery
3. **Workflow engine** - Complex multi-step workflows
4. **Human-in-the-loop** - Approval steps
5. **Fine-tuned models** - Domain-specific models
6. **Advanced routing** - Smart agent selection
7. **Observability dashboard** - Logfire integration
8. **Multi-tenancy** - Isolated agent instances
