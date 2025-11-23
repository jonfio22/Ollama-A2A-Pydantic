# A2A Multi-Agent Orchestration System

Production-grade Agent-to-Agent orchestration using Pydantic AI with local Ollama deployment.

## Features

- ✅ Local Ollama LLMs (no API costs)
- ✅ Multi-agent coordination via A2A protocol
- ✅ Type-safe agents with Pydantic
- ✅ Tool calling capabilities
- ✅ Sequential and parallel execution
- ✅ Context optimization
- ✅ Redis-backed persistence

## Quick Start

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
ollama pull llama3.1:8b
ollama pull llama3.2:3b
ollama pull qwen2.5:7b

# Install dependencies
pip install -r requirements.txt

# Start Redis
docker run -d -p 6379:6379 redis:7-alpine

# Run example
python examples/simple_orchestration.py
```

## Architecture

```
┌─────────────────┐
│  Orchestrator   │ (llama3.1:8b)
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼──┐  ┌──▼────┐
│Agent1│  │Agent2 │
└──────┘  └───────┘
```

## Project Structure

- `agents/` - Agent implementations
- `models/` - Pydantic schemas
- `storage/` - Persistence layer
- `a2a/` - A2A protocol integration
- `examples/` - Usage examples

## Documentation

See [Developer Handoff](docs/handoff.md) for detailed architecture and implementation guide.

## License

MIT
