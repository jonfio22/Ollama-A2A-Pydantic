#!/bin/bash
# Run tests with correct Python version

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

echo "Running tests with: $($PYTHON_BIN --version)"
echo ""

# Run the test
$PYTHON_BIN test_agents.py "$@"
