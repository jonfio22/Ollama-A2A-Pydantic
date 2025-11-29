# A2A Lightweight Agent Stack - Claude Code Configuration

This file optimizes Claude Code context window usage for the A2A multi-agent system.

## 🎯 Project Context

**A2A Orchestration System** - Production-grade multi-agent framework using Pydantic AI + Ollama

### Stack (14.1GB total)
- **Orchestrator**: phi3:3.8b (2.2GB - efficient reasoning)
- **Analyst**: mistral:7b (4.4GB - multilingual reasoning)
- **Coder**: deepseek-coder:6.7b (3.8GB - code specialization)
- **Validator**: llama3.2:3b (2.0GB - ultra-fast validation)
- **Vision**: moondream:1.8b (1.7GB - lightweight image analysis)

## 🚀 Quick Commands

```bash
# Setup from scratch
./scripts/setup.sh

# Pull models only
./scripts/pull-models.sh

# Run all agents (separate terminals)
uvicorn main:orchestrator_app --port 8000 --reload
uvicorn main:analyst_app --port 8001 --reload
uvicorn main:coder_app --port 8002 --reload
uvicorn main:validator_app --port 8003 --reload
uvicorn main:vision_app --port 8004 --reload

# Or with Docker
docker-compose up -d
```

## 📁 Key Files (Minimize Context Loading)

### Don't Read Unless Needed:
- `tests/` - Only when fixing test failures
- `docs/` - Reference, don't fully load
- `examples/` - Use for inspiration, not context

### Read Strategically:
- `main.py` - Agent setup (read first)
- `config/settings.py` - Configuration
- `agents/orchestrator.py` - Coordination logic
- `agents/specialists/*` - Only the one you're modifying
- `models/schemas.py` - I/O types (skim only)

### Ignore These:
- `.git/` directory
- `venv/` and virtual environments
- `__pycache__/` directories
- `.env` files (read .env.example instead)

## 🧠 Context Window Optimization

### Strategy: Grep + Targeted Reads

When exploring:
```bash
# Find function definitions
grep -r "def " agents/ --include="*.py" | head -20

# Find specific agent
grep -r "class.*Agent" agents/ --include="*.py"

# Find tool definitions
grep -r "@agent.tool" agents/ --include="*.py"

# Find imports
grep -r "from pydantic_ai" . --include="*.py" | head -10
```

### What to Include in Context

**Always Include:**
1. Current task description
2. Specific file being modified
3. Related function signatures

**Sometimes Include:**
1. Similar patterns from other agents
2. Model/schema definitions
3. Configuration values

**Never Include:**
1. Full test files (check specific tests)
2. Large JSON/data files
3. Full dependency lists
4. Historical comments

## 🔍 Common Tasks

### Adding a New Tool

1. **Check existing tools** (grep for @agent.tool)
2. **Read target agent file** (e.g., agents/specialists/analyst.py)
3. **Check schemas.py** for I/O types
4. **Implement tool** following existing patterns
5. **No need to read other agents**

### Creating New Agent

1. **Read agents/base.py** (factory patterns)
2. **Read one specialist agent** (pattern reference)
3. **Read models/dependencies.py** (dependency types)
4. **Create new agent** in agents/specialists/
5. **Register in main.py**

### Debugging Agent Calls

1. **Check main.py** (agent setup)
2. **Check config/settings.py** (model names)
3. **Check a2a/worker.py** (protocol handling)
4. **Read specific agent** file if needed
5. **Use grep** to find tool calls

### API/Integration Issues

1. **Check a2a/server.py** (FastAPI setup)
2. **Check a2a/client.py** (HTTP client)
3. **Check a2a/worker.py** (request/response)
4. **Only read relevant agent** if issue persists

## 📊 File Size Reference

Use these to estimate context impact:

```
agents/orchestrator.py        ~400 lines
agents/specialists/analyst.py ~300 lines
agents/specialists/coder.py   ~350 lines
agents/specialists/validator.py ~200 lines
agents/specialists/vision.py  ~250 lines
models/schemas.py             ~400 lines
main.py                       ~330 lines
a2a/server.py                 ~200 lines
```

**Total reading all core files** ≈ 60K tokens

## 🎯 When You Need Full Context

Only read multiple agent files when:
1. ✅ Adding inter-agent communication
2. ✅ Refactoring tool definitions
3. ✅ Changing dependency injection
4. ✅ Modifying A2A protocol

Otherwise, use **grep + targeted reads**.

## 🚀 Development Workflow

### Phase 1: Understanding (10 min)
```bash
# Read high-level overview
cat SETUP_LIGHTWEIGHT.md
cat ARCHITECTURE.md

# Grep for what you need
grep -r "class.*Orchestrator" . --include="*.py"
```

### Phase 2: Implementation (varies)
```bash
# Read only files you're modifying
# Use grep to understand patterns
# Follow existing conventions
```

### Phase 3: Testing (5 min)
```bash
# Run quick test
pytest tests/test_agents.py -v -k "test_name"

# Test full stack
./scripts/setup.sh && docker-compose up
```

## 💡 Pro Tips

1. **Use grep intelligently** - Find what you need before reading
2. **Read .example files** - Not actual .env or config
3. **Check ARCHITECTURE.md** - Before asking architecture questions
4. **Run setup.sh once** - Skip detailed reading for first-time setup
5. **Reference patterns** - Copy from existing agents, don't reinvent
6. **Use grep in editor** - Most editors have built-in grep (Ctrl+Shift+F)

## 🔗 Documentation Structure

- `README.md` - Feature overview
- `ARCHITECTURE.md` - Design patterns, extension points
- `SETUP_LIGHTWEIGHT.md` - This setup guide
- `main.py` - Application entry point (read first)

## ✨ Context-Saving Patterns

### Pattern 1: New Tool
```python
# Instead of reading full agent, grep for:
grep -A 10 "@analyst_agent.tool" agents/specialists/analyst.py

# Then copy + adapt pattern
```

### Pattern 2: New Agent
```python
# Instead of reading ARCHITECTURE, grep for:
grep "class.*Agent.*:" agents/ --include="*.py" -A 3

# Then copy from create_agent() in agents/base.py
```

### Pattern 3: Schema Changes
```python
# Don't read full models/schemas.py, grep for:
grep "class.*Output.*:" models/schemas.py -A 5

# Find similar output type, copy pattern
```

## 🎓 Learning Resources (in Order)

1. **SETUP_LIGHTWEIGHT.md** (this file) - Setup guide
2. **main.py** - Application structure
3. **agents/base.py** - Agent factory patterns
4. **One specialist agent** - Implementation reference
5. **ARCHITECTURE.md** - Deep dive (only if needed)

---

**Remember: Context is precious. Use grep before grep + read, read before reading everything.**
