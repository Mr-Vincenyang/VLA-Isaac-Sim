#!/bin/bash
# =================================================================
# VLA Server Startup Script
# =================================================================
# This script starts the VLA inference server.
#
# Usage:
#   ./start_server.sh              # Start with default settings
#   ./start_server.sh --model <model> --port <port>  # Custom settings
#
# Requirements:
#   - Python 3.8+ with pip
#   - CUDA-capable GPU (24GB+ VRAM recommended)
#   - Required packages: pip install -r server/requirements_server.txt
# =================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
MODEL="${VLA_MODEL:-openvla/openvla-7b}"
PORT="${VLA_PORT:-8000}"
QUANTIZATION="${VLA_QUANTIZATION:-}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --quantization)
            QUANTIZATION="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model <name>       VLA model name (default: openvla/openvla-7b)"
            echo "  --port <port>        Server port (default: 8000)"
            echo "  --quantization <q>   Quantization: int4, int8, or none (default: none)"
            echo ""
            echo "Environment Variables:"
            echo "  VLA_MODEL           Model name (default: openvla/openvla-7b)"
            echo "  VLA_PORT            Server port (default: 8000)"
            echo "  VLA_QUANTIZATION    Quantization option"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}VLA Server Startup${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check Python version
echo -e "${YELLOW}[1/4] Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_MAJOR=3
REQUIRED_MINOR=8
CURRENT_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
CURRENT_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$CURRENT_MAJOR" -lt "$REQUIRED_MAJOR" ] || ([ "$CURRENT_MAJOR" -eq "$REQUIRED_MAJOR" ] && [ "$CURRENT_MINOR" -lt "$REQUIRED_MINOR" ]; then
    echo -e "${RED}Error: Python $REQUIRED_MAJOR.$REQUIRED_MINOR+ required, found $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "  Python version: $PYTHON_VERSION ✓"

# Check GPU availability
echo -e "${YELLOW}[2/4] Checking GPU availability...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  GPU detected (nvidia-smi available)"
    echo -e "  GPU check passed ✓"
else
    echo -e "${YELLOW}  Warning: nvidia-smi not found. Make sure CUDA is installed.${NC}"
fi

# Install dependencies if needed
echo -e "${YELLOW}[3/4] Checking dependencies...${NC}"
if [ -f "server/requirements_server.txt" ]; then
    pip install -r server/requirements_server.txt --quiet 2>/dev/null || echo -e "${YELLOW}  Warning: Could not install dependencies (may already be installed)${NC}"
    echo -e "  Dependencies check complete ✓"
else
    echo -e "${YELLOW}  Warning: requirements_server.txt not found, skipping dependency check${NC}"
fi

# Start the server
echo -e "${YELLOW}[4/4] Starting VLA server...${NC}"
echo ""
echo -e "  Model:        ${GREEN}$MODEL${NC}"
echo -e "  Port:         ${GREEN}$PORT${NC}"
if [ -n "$QUANTIZATION" ]; then
    echo -e "  Quantization: ${GREEN}$QUANTIZATION${NC}"
fi
echo ""

# Build command
CMD="python server/server_deploy.py --model '$MODEL' --port $PORT"
if [ -n "$QUANTIZATION" ]; then
    CMD="$CMD --quantization $QUANTIZATION"
fi

echo -e "${GREEN}Starting server with command:${NC}"
echo "  $CMD"
echo ""

# Run the server
eval $CMD
