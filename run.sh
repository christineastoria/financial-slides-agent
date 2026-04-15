#!/bin/bash
# =============================================================================
# Financial Slide Generator - Datadog + LangSmith Tracing Demo
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

cd "$(dirname "$0")"

echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}  Financial Slide Generator Demo             ${NC}"
echo -e "${BLUE}  Datadog APM + LLM Obs + LangSmith         ${NC}"
echo -e "${BLUE}=============================================${NC}"
echo ""

# Load environment from parent directory
if [ -f "../.env" ]; then
    echo -e "${GREEN}Loading environment from ../.env${NC}"
    set -a
    source ../.env
    set +a
else
    echo -e "${RED}Error: ../.env not found!${NC}"
    echo "Configure your API keys in the parent directory's .env file."
    exit 1
fi

# Override service name for this demo
export DD_SERVICE=financial-slide-agent-demo
export DD_LLMOBS_ML_APP=financial-slide-agent-demo

# Ensure Playwright browsers are installed (for HTML -> PNG conversion)
echo -e "${YELLOW}Checking Playwright browsers...${NC}"

# Parse arguments
USE_DDTRACE=true
for arg in "$@"; do
    case $arg in
        --no-dd) USE_DDTRACE=false; shift ;;
    esac
done

# Check for virtual environment
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    uv venv
fi

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
uv sync
uv run playwright install chromium 2>/dev/null || echo -e "${YELLOW}Playwright chromium already installed${NC}"

# Print configuration
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "  LangSmith Project: ${LANGSMITH_PROJECT:-default}"
echo "  DD Service:        ${DD_SERVICE}"
echo "  DD Environment:    ${DD_ENV:-development}"
echo "  DD Site:           ${DD_SITE:-datadoghq.com}"
echo "  Datadog APM:       $([ "$USE_DDTRACE" = true ] && echo "Enabled" || echo "Disabled")"
echo "  LLM Observability: Enabled"
echo ""

echo -e "${GREEN}Starting server on http://localhost:8001${NC}"
echo ""

if [ "$USE_DDTRACE" = true ]; then
    echo -e "${BLUE}Running with ddtrace instrumentation...${NC}"
    echo ""
    uv run ddtrace-run uvicorn server:app \
        --host 0.0.0.0 \
        --port 8001 \
        --reload
else
    echo -e "${YELLOW}Running without Datadog tracing...${NC}"
    echo ""
    uv run uvicorn server:app \
        --host 0.0.0.0 \
        --port 8001 \
        --reload
fi
