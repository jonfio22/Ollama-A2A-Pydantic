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

# Pull models for different agents (lightweight stack)
echo ""
echo "📥 Pulling orchestrator model (phi3:3.8b)..."
ollama pull phi3:3.8b

echo ""
echo "📥 Pulling analyst model (mistral:7b)..."
ollama pull mistral:7b

echo ""
echo "📥 Pulling coder model (deepseek-coder:6.7b)..."
ollama pull deepseek-coder:6.7b

echo ""
echo "📥 Pulling validator model (llama3.2:3b)..."
ollama pull llama3.2:3b

echo ""
echo "📥 Pulling vision model (moondream:1.8b)..."
ollama pull moondream:1.8b

echo ""
echo "✅ All models pulled successfully!"
echo ""
ollama list
