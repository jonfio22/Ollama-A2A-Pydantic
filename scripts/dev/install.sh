#!/bin/bash
# Install A2A package with correct Python version

set -e

echo "🔧 Installing A2A package..."

# Set Python version
pyenv local 3.11.9

# Verify Python version
PYTHON_VERSION=$(python3 --version)
echo "Using: $PYTHON_VERSION"

if [[ ! "$PYTHON_VERSION" =~ "3.11" ]]; then
    echo "❌ Error: Need Python 3.11, but found $PYTHON_VERSION"
    echo "Run: pyenv local 3.11.9"
    exit 1
fi

# Install the package in editable mode
echo "Installing a2a package..."
python3 -m pip install -e .

echo "✅ Installation complete!"
echo ""
echo "You can now run:"
echo "  python3 test_agents.py"
echo "  python3 examples/simple_orchestration.py"
