# A2A Multi-Agent Orchestration System

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Pydantic AI](https://img.shields.io/badge/Pydantic%20AI-0.0.14+-green.svg)](https://ai.pydantic.dev/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLMs-orange.svg)](https://ollama.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Production-grade Agent-to-Agent (A2A) orchestration system using **Pydantic AI** with **local Ollama** deployment. Build sophisticated multi-agent workflows with zero API costs and complete privacy.

## ✨ Features

- 🤖 **Multi-Agent Coordination** - Orchestrator delegates to specialist agents
- 🔄 **A2A Protocol** - Standards-based agent communication
- 🚀 **Parallel Execution** - Run independent tasks concurrently
- 🛠️ **Tool Calling** - Rich tool support for each agent
- 💾 **Context Management** - Maintain conversation state across agents
- 🎯 **Type Safety** - Full Pydantic validation for all I/O
- 🏠 **Local Ollama** - No API costs, complete privacy
- 🐳 **Docker Ready** - Full containerization support

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Orchestrator Agent                       │
│              (llama3.1:8b - Coordination)                   │
└────────────┬────────────────────────────────┬───────────────┘
             │                                │
             │ A2A Protocol (HTTP/JSON-RPC)   │
             │                                │
    ┌────────▼────────┐            ┌─────────▼────────┐
    │  Data Analyst   │            │  Code Generator  │
    │  (qwen2.5:7b)   │            │ (deepseek:16b)   │
    └────────┬────────┘            └─────────┬────────┘
             │                                │
             │        ┌──────────────┐        │
             └────────►   Validator  ◄────────┘
                      │ (llama3.2:3b)│
                      └──────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Storage Layer    │
                    │ (Redis/In-Memory) │
                    └───────────────────┘
```

## 🎯 Agents

### Orchestrator Agent
- **Model**: llama3.1:8b
- **Role**: Task coordination and delegation
- **Capabilities**:
  - Analyze task complexity
  - Delegate to specialists (sequential or parallel)
  - Synthesize results from multiple agents
  - Save/retrieve intermediate results

### Data Analyst Agent
- **Model**: qwen2.5:7b (strong analytical reasoning)
- **Role**: Data analysis and statistics
- **Tools**:
  - `calculate_statistics` - Mean, median, stdev, etc.
  - `analyze_trends` - Time-series trend detection
  - `compare_datasets` - Dataset comparison
  - `cache_result` - Result caching

### Code Generator Agent
- **Model**: deepseek-coder-v2:16b (specialized for code)
- **Role**: Code generation and validation
- **Tools**:
  - `validate_syntax` - Syntax checking
  - `generate_tests` - Unit test generation
  - `find_dependencies` - Dependency extraction
  - `format_code` - Code formatting
  - `get_code_template` - Template retrieval

### Validator Agent
- **Model**: llama3.2:3b (fast validation)
- **Role**: Quality assurance and validation
- **Tools**:
  - `check_format` - Format validation (JSON, email, URL, etc.)
  - `check_length` - Length constraints
  - `check_patterns` - Regex pattern matching
  - `check_completeness` - Required field validation
  - `validate_against_schema` - Schema validation

## 🚀 Quick Start

### Prerequisites

1. **Install Ollama**
   ```bash
   # macOS
   brew install ollama

   # Linux
   curl -fsSL https://ollama.com/install.sh | sh

   # Windows - Download from https://ollama.com/download
   ```

2. **Start Ollama**
   ```bash
   ollama serve
   ```

### Automated Setup

```bash
# Run the setup script (pulls models, creates venv, installs deps)
./scripts/setup.sh

# Activate virtual environment
source venv/bin/activate
```

### Manual Setup

```bash
# Pull models
ollama pull llama3.1:8b       # Orchestrator
ollama pull qwen2.5:7b        # Analyst
ollama pull deepseek-coder-v2:16b  # Coder
ollama pull llama3.2:3b       # Validator

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
```

### Running the Agents

**Option 1: Individual Terminals**
```bash
# Terminal 1 - Orchestrator
uvicorn main:orchestrator_app --port 8000

# Terminal 2 - Analyst
uvicorn main:analyst_app --port 8001

# Terminal 3 - Coder
uvicorn main:coder_app --port 8002

# Terminal 4 - Validator
uvicorn main:validator_app --port 8003
```

**Option 2: Docker Compose**
```bash
docker-compose up
```

This will:
- Start Ollama service
- Pull all required models
- Start Redis for persistence
- Launch all 4 agents

### Running Examples

```bash
# Simple orchestration
python examples/simple_orchestration.py

# Parallel execution demo
python examples/parallel_execution.py
```

## 📖 Usage Examples

### Direct Agent Access

```python
from a2a.client import A2AClient

# Connect to a specialist agent
async with A2AClient("http://localhost:8001") as client:
    # Send analysis task
    response = await client.send_message(
        message="Calculate mean and median of [1, 2, 3, 4, 5]"
    )

    print(response["result"]["output"])
```

### Orchestrated Workflow

```python
from a2a.client import A2AClient

async with A2AClient("http://localhost:8000") as orchestrator:
    # Send complex task to orchestrator
    response = await orchestrator.send_message(
        message="""
        Analyze this dataset: [10, 20, 30, 40, 50]
        Then generate Python code to visualize it.
        Finally validate the code syntax.
        """
    )

    # Orchestrator delegates to specialists automatically
    results = response["result"]["output"]["task_results"]
    synthesis = response["result"]["output"]["synthesis"]
```

### Parallel Execution

```python
async with A2AClient("http://localhost:8000") as orchestrator:
    # Explicitly request parallel execution
    response = await orchestrator.send_message(
        message="""
        Perform these independent tasks in parallel:
        1. ANALYST: analyze [1, 2, 3, 4, 5]
        2. CODER: generate a hello world function
        3. VALIDATOR: validate email test@example.com
        """
    )
```

## 🧪 Testing

```bash
# Run unit tests
pytest tests/ -v

# Run specific test
pytest tests/test_agents.py::test_analyst_agent -v

# With coverage
pytest --cov=. --cov-report=html
```

## 🛠️ Development

### Project Structure

```
a2a-orchestration/
├── agents/              # Agent implementations
│   ├── base.py         # Base agent utilities
│   ├── orchestrator.py # Orchestrator agent
│   └── specialists/    # Specialist agents
│       ├── analyst.py  # Data analyst
│       ├── coder.py    # Code generator
│       └── validator.py # Validator
├── models/             # Pydantic models
│   ├── schemas.py      # I/O schemas
│   └── dependencies.py # Dependency injection
├── storage/            # Persistence layer
│   ├── interfaces.py   # Abstract interfaces
│   ├── redis_storage.py # Redis implementation
│   └── memory_storage.py # In-memory (testing)
├── a2a/                # A2A protocol
│   ├── worker.py       # Agent worker
│   ├── client.py       # A2A client
│   └── server.py       # Server setup
├── config/             # Configuration
│   └── settings.py     # Settings management
├── examples/           # Example scripts
├── tests/              # Test suite
├── scripts/            # Utility scripts
├── main.py             # Application entry
└── docker-compose.yml  # Docker setup
```

### Adding a New Agent

1. **Define the agent's output schema** in `models/schemas.py`:
   ```python
   class MyAgentOutput(BaseModel):
       result: str
       confidence: float
   ```

2. **Create dependencies** in `models/dependencies.py`:
   ```python
   @dataclass
   class MyAgentDependencies:
       agent_id: str
       storage: StorageInterface
   ```

3. **Implement the agent** in `agents/specialists/`:
   ```python
   from agents.base import create_agent

   my_agent = create_agent(
       model='ollama:llama3.1:8b',
       agent_id='my-agent',
       instructions='...',
       deps_type=MyAgentDependencies,
       output_type=MyAgentOutput
   )

   @my_agent.tool
   async def my_tool(ctx, param: str) -> str:
       # Tool implementation
       return "result"
   ```

4. **Register in main.py** and expose as A2A service

### Adding Tools to Agents

```python
from pydantic_ai import RunContext

@agent.tool
async def my_tool(
    ctx: RunContext[MyDependencies],
    param1: str,
    param2: int
) -> Dict[str, Any]:
    """
    Tool description (visible to LLM).

    Args:
        ctx: Agent context with dependencies
        param1: Parameter description
        param2: Parameter description

    Returns:
        Result dictionary
    """
    # Access dependencies
    storage = ctx.deps.storage

    # Tool logic
    result = do_something(param1, param2)

    return {"result": result}
```

## 🐳 Docker Deployment

### Build and Run

```bash
# Build images
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Environment Variables

Configure in `.env` or docker-compose.yml:

```bash
OLLAMA_BASE_URL=http://ollama:11434
REDIS_URL=redis://redis:6379
ORCHESTRATOR_MODEL=ollama:llama3.1:8b
ANALYST_MODEL=ollama:qwen2.5:7b
CODER_MODEL=ollama:deepseek-coder-v2:16b
FAST_MODEL=ollama:llama3.2:3b
```

## 📊 Performance

### Model Selection

| Agent | Model | Size | Speed | Use Case |
|-------|-------|------|-------|----------|
| Orchestrator | llama3.1:8b | 4.7GB | ⚡⚡ | Complex reasoning |
| Analyst | qwen2.5:7b | 4.7GB | ⚡⚡ | Data analysis |
| Coder | deepseek-coder-v2:16b | 9GB | ⚡ | Code generation |
| Validator | llama3.2:3b | 2GB | ⚡⚡⚡ | Fast validation |

### Parallel vs Sequential

Parallel execution can provide significant speedup for independent tasks:

```
Sequential: Task1 → Task2 → Task3 (9 seconds)
Parallel:   Task1 ⎱ Task2 ⎰ Task3 (3 seconds)
Speedup: 3x faster
```

## 🔧 Configuration

### Ollama Settings

```bash
# Increase context window
export OLLAMA_MAX_CONTEXT=8192

# Increase parallel requests
export OLLAMA_MAX_LOADED_MODELS=4

# GPU memory allocation
export OLLAMA_GPU_MEMORY=16GB

# Keep models in memory
export OLLAMA_KEEP_ALIVE=1h
```

### Agent Configuration

Modify models in `.env`:
```bash
# Use different models
ORCHESTRATOR_MODEL=ollama:llama3.1:70b  # More powerful
ANALYST_MODEL=ollama:qwen2.5:14b       # More capable
```

## 🐛 Troubleshooting

### Ollama Not Running

```bash
# Check if running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve
```

### Models Not Found

```bash
# List installed models
ollama list

# Pull missing model
ollama pull llama3.1:8b
```

### Slow Performance

- Use smaller models (llama3.2:3b instead of llama3.1:8b)
- Enable GPU acceleration
- Increase `OLLAMA_KEEP_ALIVE` to keep models loaded
- Use parallel execution when possible

### Connection Errors

```bash
# Check if agents are running
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health
```

## 📚 Documentation

- [Developer Handoff](docs/handoff.md) - Complete implementation guide
- [A2A Protocol](https://github.com/a2aproject/A2A) - Protocol specification
- [Pydantic AI](https://ai.pydantic.dev/) - Agent framework docs
- [Ollama](https://github.com/ollama/ollama) - Local LLM runtime

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **Pydantic AI** - Production-ready agent framework
- **Ollama** - Local LLM runtime
- **Google A2A Protocol** - Agent interoperability standard
- **FastAPI** - Modern async web framework

## 🌟 Features Roadmap

- [ ] Streaming responses
- [ ] WebSocket support for real-time updates
- [ ] Agent discovery and registry
- [ ] Human-in-the-loop approvals
- [ ] Fine-tuned specialist models
- [ ] Observability dashboard (Logfire integration)
- [ ] Kubernetes deployment templates
- [ ] CLI tool for agent management

## 📧 Support

For questions or issues:
- Open an issue on GitHub
- Check the troubleshooting guide
- Review the developer handoff document

---

**Built with ❤️ using Pydantic AI and Ollama**

*Zero API costs. Complete privacy. Full control.*
