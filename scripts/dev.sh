#!/bin/bash

# A2A Development Server - Runs all services
# Usage: ./scripts/dev.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════${NC}"
echo -e "${BLUE}  A2A Development Server${NC}"
echo -e "${BLUE}═══════════════════════════════════════${NC}"

# Kill any existing processes on our ports
cleanup() {
  echo -e "\n${YELLOW}Shutting down services...${NC}"

  for port in 3000 8000 8001 8002 8003 8004; do
    if lsof -ti:$port > /dev/null 2>&1; then
      echo "Killing process on port $port..."
      lsof -ti:$port | xargs kill -9 2>/dev/null || true
    fi
  done

  echo -e "${GREEN}Cleanup complete${NC}"
}

trap cleanup EXIT

# Kill existing processes
cleanup

sleep 1

echo -e "\n${BLUE}Starting services...${NC}\n"

# Start frontend (Next.js) in background
echo -e "${YELLOW}Starting Dashboard (port 3000)...${NC}"
cd "$PROJECT_ROOT/dashboard"
npm run dev > "$PROJECT_ROOT/.logs/dashboard.log" 2>&1 &
DASHBOARD_PID=$!
echo -e "${GREEN}✓ Dashboard started (PID: $DASHBOARD_PID)${NC}"

sleep 2

# Start backend agents in background
cd "$PROJECT_ROOT"

echo -e "${YELLOW}Starting Orchestrator Agent (port 8000)...${NC}"
uvicorn main:orchestrator_app --port 8000 --reload > "$PROJECT_ROOT/.logs/orchestrator.log" 2>&1 &
ORCHESTRATOR_PID=$!
echo -e "${GREEN}✓ Orchestrator started (PID: $ORCHESTRATOR_PID)${NC}"

echo -e "${YELLOW}Starting Analyst Agent (port 8001)...${NC}"
uvicorn main:analyst_app --port 8001 --reload > "$PROJECT_ROOT/.logs/analyst.log" 2>&1 &
ANALYST_PID=$!
echo -e "${GREEN}✓ Analyst started (PID: $ANALYST_PID)${NC}"

echo -e "${YELLOW}Starting Coder Agent (port 8002)...${NC}"
uvicorn main:coder_app --port 8002 --reload > "$PROJECT_ROOT/.logs/coder.log" 2>&1 &
CODER_PID=$!
echo -e "${GREEN}✓ Coder started (PID: $CODER_PID)${NC}"

echo -e "${YELLOW}Starting Validator Agent (port 8003)...${NC}"
uvicorn main:validator_app --port 8003 --reload > "$PROJECT_ROOT/.logs/validator.log" 2>&1 &
VALIDATOR_PID=$!
echo -e "${GREEN}✓ Validator started (PID: $VALIDATOR_PID)${NC}"

echo -e "${YELLOW}Starting Vision Agent (port 8004)...${NC}"
uvicorn main:vision_app --port 8004 --reload > "$PROJECT_ROOT/.logs/vision.log" 2>&1 &
VISION_PID=$!
echo -e "${GREEN}✓ Vision started (PID: $VISION_PID)${NC}"

sleep 2

echo -e "\n${GREEN}═══════════════════════════════════════${NC}"
echo -e "${GREEN}  All services running!${NC}"
echo -e "${GREEN}═══════════════════════════════════════${NC}\n"

echo -e "${BLUE}Services:${NC}"
echo -e "  ${BLUE}Dashboard${NC}       http://localhost:3000"
echo -e "  ${BLUE}Orchestrator${NC}    http://localhost:8000"
echo -e "  ${BLUE}Analyst${NC}         http://localhost:8001"
echo -e "  ${BLUE}Coder${NC}           http://localhost:8002"
echo -e "  ${BLUE}Validator${NC}       http://localhost:8003"
echo -e "  ${BLUE}Vision${NC}          http://localhost:8004"

echo -e "\n${BLUE}Logs:${NC}"
echo -e "  Dashboard    : .logs/dashboard.log"
echo -e "  Orchestrator : .logs/orchestrator.log"
echo -e "  Analyst      : .logs/analyst.log"
echo -e "  Coder        : .logs/coder.log"
echo -e "  Validator    : .logs/validator.log"
echo -e "  Vision       : .logs/vision.log"

echo -e "\n${YELLOW}Press Ctrl+C to stop all services${NC}\n"

# Wait for all background processes
wait
