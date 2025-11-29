# 🚀 Getting Started - A2A Lightweight Stack

**⏱️ Estimated Setup Time: 5 minutes** (first run includes ~10GB model downloads)

## 🎯 What You're Setting Up

A production-grade **5-agent orchestration system** with:
- ✅ **14.7GB total size** (8% smaller than v1)
- ✅ **Zero API costs** (runs locally)
- ✅ **Full privacy** (no cloud services)
- ✅ **Multi-specialist agents** (analysis, coding, validation, vision)
- ✅ **Parallel execution** (concurrent task handling)

## 📋 Checklist

- [ ] **Ollama installed** (brew install ollama / download)
- [ ] **4GB+ RAM available** (8GB+ recommended)
- [ ] **macOS/Linux/Windows (WSL2)**
- [ ] **15GB disk space** for models

## ⚡ 5-Minute Quickstart

### Step 1: Install Ollama (if needed)

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download from https://ollama.com/download
```

### Step 2: Start Ollama

```bash
ollama serve
# Keep this running in the background
```

### Step 3: Clone & Setup

```bash
cd /path/to/a2a-nov

# Run one-line setup (pulls models, creates venv, installs deps)
chmod +x scripts/setup.sh
./scripts/setup.sh

# Takes ~2-5 minutes depending on internet
# Models download: ~10GB
```

### Step 4: Start Agents

**Option A: macOS/Linux (Easy - one command)**

```bash
chmod +x scripts/start-dev.sh
./scripts/start-dev.sh
# Opens tmux with all 5 agents running
```

**Option B: Multiple terminals (Compatible with all OS)**

```bash
# Terminal 1: Orchestrator
source venv/bin/activate
uvicorn main:orchestrator_app --port 8000 --reload

# Terminal 2: Analyst
source venv/bin/activate
uvicorn main:analyst_app --port 8001 --reload

# Terminal 3: Coder
source venv/bin/activate
uvicorn main:coder_app --port 8002 --reload

# Terminal 4: Validator
source venv/bin/activate
uvicorn main:validator_app --port 8003 --reload

# Terminal 5: Vision
source venv/bin/activate
uvicorn main:vision_app --port 8004 --reload
```

**Option C: Docker (Recommended for production)**

```bash
docker-compose up -d
# All services start automatically
```

### Step 5: Verify Setup

```bash
# All should return {"status": "ok"}
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health
curl http://localhost:8004/health
```

## ✨ First Test

### Via cURL

```bash
# Simple analysis task
curl -X POST http://localhost:8001/a2a/run \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "run",
    "params": {
      "message": "Calculate mean, median, and standard deviation of [1, 2, 3, 4, 5]"
    }
  }' | jq .
```

### Via Python

```python
import asyncio
from a2a.client import A2AClient

async def test():
    async with A2AClient("http://localhost:8000") as client:
        response = await client.send_message(
            message="Analyze this dataset and generate visualization code: [10, 20, 30, 40, 50]"
        )
        print(response["result"]["output"])

asyncio.run(test())
```

## 📊 Agent Stack

| Agent | Port | Model | Size | Use |
|-------|------|-------|------|-----|
| 🎯 **Orchestrator** | 8000 | phi-3.5-mini | 2.5GB | Task coordination |
| 📈 **Analyst** | 8001 | mistral | 4.5GB | Data analysis |
| 💻 **Coder** | 8002 | deepseek-coder | 4.5GB | Code generation |
| ✅ **Validator** | 8003 | llama3.2 | 2GB | Quality checks |
| 👁️ **Vision** | 8004 | moondream | 1.2GB | Image analysis |

## 🆘 Troubleshooting

### "Ollama not running"
```bash
# Check status
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve
```

### "Models not found"
```bash
# Re-run setup to pull models
./scripts/setup.sh

# Or manually pull
ollama pull phi-3.5-mini:3.8b
ollama pull mistral:7b
ollama pull deepseek-coder:7b
ollama pull llama3.2:3b
ollama pull moondream:1.8b
```

### "Port already in use"
```bash
# Find and kill process
lsof -i :8000
kill -9 <PID>

# Or change port
uvicorn main:orchestrator_app --port 9000
```

### "Slow inference"
1. **GPU not enabled?** Check CUDA/Metal/ROCm
2. **Keep models loaded:** `export OLLAMA_KEEP_ALIVE=1h`
3. **Too many models loaded?** Close other applications
4. **Use smaller models?** See configuration guide

## 🎓 Next Steps

### 1. Explore the System

```bash
# Read the setup guide
cat SETUP_LIGHTWEIGHT.md

# Understand the architecture
cat ARCHITECTURE.md

# Run examples
python examples/simple_orchestration.py
python examples/parallel_execution.py
```

### 2. Make Your First Request

The orchestrator accepts complex tasks and delegates to specialists:

```bash
curl -X POST http://localhost:8000/a2a/run \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "req1",
    "method": "run",
    "params": {
      "message": "I have sales data [100, 150, 200, 180, 220]. Analyze it and generate Python code to visualize it."
    }
  }' | jq .
```

### 3. Build Your Own Agent

See [ARCHITECTURE.md](./ARCHITECTURE.md) for extending with custom agents.

## 📚 Documentation

| Doc | Purpose |
|-----|---------|
| **README.md** | Feature overview & examples |
| **SETUP_LIGHTWEIGHT.md** | Complete setup + troubleshooting |
| **ARCHITECTURE.md** | Design patterns & extension |
| **GETTING_STARTED.md** | This file - quick reference |
| **.claude/claude.md** | Claude Code optimization |

## 🔗 Quick Links

- **Health Check**: http://localhost:8000/health
- **Orchestrator Docs**: http://localhost:8000/.well-known/agent.json
- **Local Ollama**: http://localhost:11434/api/tags

## ⚙️ Configuration

### Environment Variables

Edit `.env` to customize models or ports:

```bash
ORCHESTRATOR_MODEL=ollama:phi-3.5-mini:3.8b
ANALYST_MODEL=ollama:mistral:7b
CODER_MODEL=ollama:deepseek-coder:7b
FAST_MODEL=ollama:llama3.2:3b
VISION_MODEL=ollama:moondream:1.8b
```

### Performance Tuning

```bash
# Before running ollama serve, set:
export OLLAMA_KEEP_ALIVE=1h        # Keep models in memory
export OLLAMA_MAX_LOADED_MODELS=5  # Load all 5 simultaneously
export OLLAMA_MAX_CONTEXT=8192     # Larger context window
```

## 🐛 Debug Mode

```bash
# Verbose logging
export LOG_LEVEL=DEBUG

# Run with reload
uvicorn main:orchestrator_app --port 8000 --reload --log-level debug
```

## 📊 Monitoring

### Check Loaded Models

```bash
curl http://localhost:11434/api/tags | jq '.models[] | .name'
```

### View Agent Capabilities

```bash
curl http://localhost:8000/.well-known/agent.json | jq '.capabilities'
```

### Memory Usage (macOS)

```bash
# See which processes use memory
top -o MEM

# Focus on python/ollama
ps aux | grep -E "ollama|python"
```

## 🎯 Common Patterns

### Pattern 1: Data Analysis + Code Generation

```python
message = """
Analyze this sales data: [100, 150, 200, 180, 220]
Generate Python code to:
1. Calculate statistics
2. Create visualization
3. Identify trends
"""

# Orchestrator automatically delegates to:
# 1. Analyst for statistical analysis
# 2. Coder for Python code generation
# 3. Validator to check code quality
```

### Pattern 2: Validation Pipeline

```python
message = """
Validate this JSON: {"name": "John", "age": 30, "email": "john@example.com"}
Check:
1. Required fields present
2. Email format valid
3. Age is reasonable
"""

# Validator agent handles all checks
```

### Pattern 3: Vision + Analysis

```python
message = """
Analyze this image (base64): [image_data]
1. Extract text (OCR)
2. Describe content
3. Generate image generation prompt
"""

# Vision agent processes, Analyst interprets
```

## 💡 Tips & Tricks

1. **Fast iteration**: Use `--reload` flag for auto-restart on code changes
2. **Parallel tasks**: Orchestrator automatically runs independent tasks in parallel
3. **Caching**: Tools can cache results to avoid re-computation
4. **GPU boost**: Enable CUDA/Metal for 2-4x faster inference
5. **Model swaps**: Change models in `.env` without restarting

## 🆘 Getting Help

1. **Setup issues**: See SETUP_LIGHTWEIGHT.md troubleshooting
2. **Architecture questions**: Read ARCHITECTURE.md
3. **Agent issues**: Check main.py and target agent file
4. **Ollama problems**: See Ollama docs at ollama.com

## 🎉 You're Ready!

Everything is set up. Start building with the A2A system:

```bash
# Next: Run your first test
curl http://localhost:8000/health

# Then: Explore examples
python examples/simple_orchestration.py
```

---

**Questions?** Check the docs or run with `--log-level debug` for details.

**Ready to extend?** See ARCHITECTURE.md for adding custom agents and tools.

