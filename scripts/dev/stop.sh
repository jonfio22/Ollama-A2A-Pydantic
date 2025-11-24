#!/bin/bash
# Stop all A2A agents

echo "Stopping all A2A agents..."

# Kill processes on agent ports
for port in 8000 8001 8002 8003; do
    pids=$(lsof -ti:$port 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "Killing processes on port $port: $pids"
        kill -9 $pids 2>/dev/null
    fi
done

# Also kill any remaining python processes running start_all.py
pkill -f "start_all.py" 2>/dev/null

echo "✓ All agents stopped"
