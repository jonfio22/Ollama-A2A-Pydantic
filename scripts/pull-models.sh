#!/bin/bash
set -e

echo "🤖 Pulling Ollama models for A2A agents..."

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama service..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
  if curl -s http://ollama:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama is ready!"
    break
  fi
  attempt=$((attempt + 1))
  echo "   Attempt $attempt/$max_attempts..."
  sleep 2
done

if [ $attempt -eq $max_attempts ]; then
  echo "❌ Ollama failed to start in time"
  exit 1
fi

# Pull models for different agents
echo ""
echo "📥 Pulling orchestrator model (llama3.1:8b)..."
ollama pull llama3.1:8b

echo ""
echo "📥 Pulling analyst model (qwen2.5:7b)..."
ollama pull qwen2.5:7b

echo ""
echo "📥 Pulling coder model (deepseek-coder-v2:16b)..."
ollama pull deepseek-coder-v2:16b

echo ""
echo "📥 Pulling fast model for validator (llama3.2:3b)..."
ollama pull llama3.2:3b

echo ""
echo "✅ All models pulled successfully!"
echo ""
ollama list
