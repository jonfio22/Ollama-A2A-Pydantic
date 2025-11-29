#!/bin/bash
# Development startup script for A2A Multi-Agent System
# Starts all 5 agents in tmux windows for easy management

set -e

echo "🚀 A2A Agent Startup Script"
echo "============================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Run: ./scripts/setup.sh"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if Ollama is running
echo "🔍 Checking Ollama..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Ollama is not running"
    echo "   Start with: ollama serve"
    exit 1
fi
echo "✅ Ollama is running"
echo ""

# Check if tmux is available
if ! command -v tmux &> /dev/null; then
    echo "⚠️  tmux not found - using separate approach"
    echo ""
    echo "📋 Run each command in a separate terminal:"
    echo ""
    echo "Terminal 1 (Orchestrator):"
    echo "  uvicorn main:orchestrator_app --port 8000 --reload"
    echo ""
    echo "Terminal 2 (Analyst):"
    echo "  uvicorn main:analyst_app --port 8001 --reload"
    echo ""
    echo "Terminal 3 (Coder):"
    echo "  uvicorn main:coder_app --port 8002 --reload"
    echo ""
    echo "Terminal 4 (Validator):"
    echo "  uvicorn main:validator_app --port 8003 --reload"
    echo ""
    echo "Terminal 5 (Vision):"
    echo "  uvicorn main:vision_app --port 8004 --reload"
    echo ""
    exit 0
fi

# Create or attach to tmux session
SESSION="a2a-agents"

# Kill existing session if it exists
if tmux has-session -t $SESSION 2>/dev/null; then
    echo "📋 Found existing session '$SESSION'"
    echo "   Kill with: tmux kill-session -t a2a-agents"
    echo "   Or attach: tmux attach -t a2a-agents"
    exit 0
fi

# Create new session
echo "🔧 Creating tmux session..."
tmux new-session -d -s $SESSION -x 200 -y 50

# Orchestrator Agent (port 8000)
tmux new-window -t $SESSION:0 -n "Orchestrator"
tmux send-keys -t $SESSION:0 "cd /Users/fiorante/Documents/a2a-nov && source venv/bin/activate && uvicorn main:orchestrator_app --port 8000 --reload" Enter

# Analyst Agent (port 8001)
tmux new-window -t $SESSION:1 -n "Analyst"
tmux send-keys -t $SESSION:1 "cd /Users/fiorante/Documents/a2a-nov && source venv/bin/activate && uvicorn main:analyst_app --port 8001 --reload" Enter

# Coder Agent (port 8002)
tmux new-window -t $SESSION:2 -n "Coder"
tmux send-keys -t $SESSION:2 "cd /Users/fiorante/Documents/a2a-nov && source venv/bin/activate && uvicorn main:coder_app --port 8002 --reload" Enter

# Validator Agent (port 8003)
tmux new-window -t $SESSION:3 -n "Validator"
tmux send-keys -t $SESSION:3 "cd /Users/fiorante/Documents/a2a-nov && source venv/bin/activate && uvicorn main:validator_app --port 8003 --reload" Enter

# Vision Agent (port 8004)
tmux new-window -t $SESSION:4 -n "Vision"
tmux send-keys -t $SESSION:4 "cd /Users/fiorante/Documents/a2a-nov && source venv/bin/activate && uvicorn main:vision_app --port 8004 --reload" Enter

# Health check window
tmux new-window -t $SESSION:5 -n "Health"
tmux send-keys -t $SESSION:5 "sleep 3 && echo '🏥 Waiting for agents to start...' && sleep 2" Enter

sleep 2

# Verify agents are starting
echo ""
echo "⏳ Waiting for agents to start (3 seconds)..."
sleep 3

echo ""
echo "🏥 Checking agent health..."
echo ""

agents=(
    "http://localhost:8000 Orchestrator"
    "http://localhost:8001 Analyst"
    "http://localhost:8002 Coder"
    "http://localhost:8003 Validator"
    "http://localhost:8004 Vision"
)

all_healthy=true
for agent_info in "${agents[@]}"; do
    url=$(echo $agent_info | cut -d' ' -f1)
    name=$(echo $agent_info | cut -d' ' -f2-)

    if curl -s "$url/health" > /dev/null 2>&1; then
        echo "✅ $name is ready"
    else
        echo "⏳ $name is starting..."
        all_healthy=false
    fi
done

echo ""
echo "════════════════════════════════════════════════"
echo "✅ Agents Started!"
echo "════════════════════════════════════════════════"
echo ""
echo "📊 Ports:"
echo "   🎯 Orchestrator: http://localhost:8000"
echo "   📈 Analyst:      http://localhost:8001"
echo "   💻 Coder:        http://localhost:8002"
echo "   ✅ Validator:    http://localhost:8003"
echo "   👁️  Vision:       http://localhost:8004"
echo ""
echo "🎮 Tmux Controls:"
echo "   View all windows: tmux list-windows -t $SESSION"
echo "   Attach session:   tmux attach -t $SESSION"
echo "   Switch window:    tmux select-window -t $SESSION:0"
echo "   Kill session:     tmux kill-session -t $SESSION"
echo ""
echo "📝 Tmux Keys (when attached):"
echo "   Ctrl-b c         Create new window"
echo "   Ctrl-b n         Next window"
echo "   Ctrl-b p         Previous window"
echo "   Ctrl-b [         Scroll mode (q to exit)"
echo "   Ctrl-b d         Detach from session"
echo ""
echo "🧪 Test the stack:"
echo "   curl http://localhost:8000/health"
echo ""
echo "Attaching to session..."
sleep 1
tmux attach-session -t $SESSION
