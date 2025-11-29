#!/bin/bash
set -e

echo "🚀 A2A Multi-Agent Orchestration System - Setup"
echo "================================================"
echo ""

# Check if Ollama is installed
echo "1️⃣  Checking Ollama installation..."
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed"
    echo "📥 Please install Ollama first:"
    echo "   - macOS: brew install ollama"
    echo "   - Linux: curl -fsSL https://ollama.com/install.sh | sh"
    echo "   - Windows: Download from https://ollama.com/download"
    exit 1
fi
echo "✅ Ollama is installed"

# Check if Ollama is running
echo ""
echo "2️⃣  Checking if Ollama is running..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama is running"
else
    echo "⚠️  Ollama is not running"
    echo "🔧 Starting Ollama..."
    ollama serve &
    sleep 5
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama started successfully"
    else
        echo "❌ Failed to start Ollama"
        exit 1
    fi
fi

# Pull required models
echo ""
echo "3️⃣  Pulling required models..."
echo "   This may take a while depending on your internet connection."
echo ""

models=("phi3:3.8b" "mistral:7b" "deepseek-coder:6.7b" "llama3.2:3b" "moondream:1.8b")

for model in "${models[@]}"; do
    echo "📥 Pulling $model..."
    ollama pull "$model"
done

echo ""
echo "✅ All models pulled successfully!"

# Create virtual environment
echo ""
echo "4️⃣  Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "ℹ️  Virtual environment already exists"
fi

# Activate virtual environment and install dependencies
echo ""
echo "5️⃣  Installing Python dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Dependencies installed"

# Create .env file if it doesn't exist
echo ""
echo "6️⃣  Setting up environment configuration..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ .env file created from .env.example"
    echo "ℹ️  You can customize .env if needed"
else
    echo "ℹ️  .env file already exists"
fi

# Check if Redis is available (optional)
echo ""
echo "7️⃣  Checking Redis (optional for persistence)..."
if command -v redis-cli &> /dev/null; then
    if redis-cli ping > /dev/null 2>&1; then
        echo "✅ Redis is running"
    else
        echo "⚠️  Redis is installed but not running"
        echo "ℹ️  Start with: redis-server"
        echo "ℹ️  Or use Docker: docker run -d -p 6379:6379 redis:7-alpine"
    fi
else
    echo "ℹ️  Redis not installed (using in-memory storage)"
    echo "ℹ️  For persistence, install Redis or run with Docker"
fi

echo ""
echo "════════════════════════════════════════════════"
echo "✅ Setup Complete!"
echo "════════════════════════════════════════════════"
echo ""
echo "🚀 Next Steps:"
echo ""
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Start the agents (in separate terminals):"
echo "   Terminal 1: uvicorn main:orchestrator_app --port 8000"
echo "   Terminal 2: uvicorn main:analyst_app --port 8001"
echo "   Terminal 3: uvicorn main:coder_app --port 8002"
echo "   Terminal 4: uvicorn main:validator_app --port 8003"
echo "   Terminal 5: uvicorn main:vision_app --port 8004"
echo ""
echo "3. Or use Docker Compose:"
echo "   docker-compose up"
echo ""
echo "4. Run examples:"
echo "   python examples/simple_orchestration.py"
echo "   python examples/parallel_execution.py"
echo ""
echo "📖 For more info, see README.md"
echo ""
