#!/bin/bash
# A2A Multi-Agent Startup Script
# Ensures pyenv is properly initialized before starting agents

set -e

# Add Homebrew to PATH
export PATH="/opt/homebrew/bin:$PATH"

# Initialize pyenv if available
if command -v pyenv >/dev/null 2>&1; then
    eval "$(pyenv init -)"
    pyenv local 3.11.9
    PYTHON_BIN="$(pyenv which python3)"
else
    # Fallback to direct path
    PYTHON_BIN="/Users/fiorante/.pyenv/versions/3.11.9/bin/python3"
fi
echo "Starting A2A Multi-Agent System..."
echo "Using Python: $($PYTHON_BIN --version)"
echo "Python path: $PYTHON_BIN"
echo ""

# Install a2a package if not already installed
if ! $PYTHON_BIN -c "import a2a" 2>/dev/null; then
    echo "📦 Installing a2a package..."
    $PYTHON_BIN -m pip install -e . --quiet
    echo "✅ Package installed"
    echo ""
fi

# Run the startup script
$PYTHON_BIN start_all.py
