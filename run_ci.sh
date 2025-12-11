#!/bin/bash
set -e

echo "🔍 Starting CI Pipeline..."

# Check if we are running in an environment with dependencies (Docker) or Host
if ! command -v ruff &> /dev/null; then
    echo "⚠️  'ruff' command not found. You seem to be running on the Host machine."
    echo "🐳 Auto-launching CI inside 'trader' Docker container..."
    echo "=================================================="
    
    # Check if docker is available
    if ! command -v docker &> /dev/null; then
        echo "❌ Error: Docker is not installed or not in PATH."
        exit 1
    fi

    # Ensure image is built (in case dependencies changed)
    echo "🔨 Building/Updating Docker image..."
    docker compose build trader || { echo "❌ Docker build failed"; exit 1; }

    # Run this script inside the container
    echo "🚀 Executing ./run_ci.sh inside Docker..."
    docker compose run --rm trader ./run_ci.sh
    exit $?
fi

echo "=================================================="
echo "🛠️  Step 1: Linting (Ruff)"
echo "=================================================="
ruff check .
echo "✅ Linting Passed"

echo "=================================================="
echo "🛡️  Step 2: Type Checking (Mypy)"
echo "=================================================="
mypy .
echo "✅ Type Checking Passed"

echo "=================================================="
echo "🧪 Step 3: Unit Tests (Pytest)"
echo "=================================================="
# Run all tests using pytest
pytest tests/
echo "✅ All Tests Passed"

echo "🎉 CI Pipeline COMPLETE: READY FOR DEPLOY"
