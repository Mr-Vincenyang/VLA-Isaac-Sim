#!/bin/bash
# =================================================================
# VLA Local Startup Script (Isaac Sim)
# =================================================================
# This script starts VLA demos locally using Isaac Sim.
#
# Usage:
#   ./start_local.sh --demo motion           # Run motion control demo
#   ./start_local.sh --demo grasp --record   # Run grasp demo with video
#   ./start_local.sh --demo grasp --server http://localhost:8000  # With VLA server
#
# Options:
#   --demo <name>       Demo to run: motion, grasp (required)
#   --record            Enable video recording (saves to output/)
#   --server <url>      VLA server URL (optional)
#   --episodes <n>      Number of episodes for grasp demo (default: 3)
#   --isaac-path <path> Isaac Sim installation path (default: ~/isaac-sim)
#   --headless          Run in headless mode (no GUI, for video recording)
#   --help              Show this help message
#
# Environment Variables:
#   ISAAC_SIM_PATH      Isaac Sim installation path
#   VLA_SERVER_URL      VLA server URL
# =================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DEMO=""
RECORD=false
SERVER_URL=""
EPISODES=3
HEADLESS=false
ISAAC_PATH="${ISAAC_SIM_PATH:-$HOME/isaac-sim}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --demo)
            DEMO="$2"
            shift 2
            ;;
        --record)
            RECORD=true
            shift
            ;;
        --server)
            SERVER_URL="$2"
            shift 2
            ;;
        --episodes)
            EPISODES="$2"
            shift 2
            ;;
        --headless)
            HEADLESS=true
            shift
            ;;
        --isaac-path)
            ISAAC_PATH="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --demo <name>       Demo to run: motion, grasp"
            echo "  --record            Enable video recording"
            echo "  --server <url>      VLA server URL"
            echo "  --episodes <n>      Number of episodes (default: 3)"
            echo "  --headless          Run in headless mode"
            echo "  --isaac-path <path> Isaac Sim path (default: ~/isaac-sim)"
            echo ""
            echo "Examples:"
            echo "  $0 --demo motion --record"
            echo "  $0 --demo grasp --server http://localhost:8000"
            echo "  $0 --demo grasp --record --headless"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate arguments
if [ -z "$DEMO" ]; then
    echo -e "${RED}Error: --demo is required${NC}"
    echo "Use --help for usage information"
    exit 1
fi

if [ "$DEMO" != "motion" ] && [ "$DEMO" != "grasp" ]; then
    echo -e "${RED}Error: Invalid demo: $DEMO${NC}"
    echo "Valid demos: motion, grasp"
    exit 1
fi

# Check if Isaac Sim path exists
if [ ! -d "$ISAAC_PATH" ]; then
    echo -e "${RED}Error: Isaac Sim not found at: $ISAAC_PATH${NC}"
    echo "Use --isaac-path to specify the correct path"
    exit 1
fi

# Check for python.sh
if [ ! -f "$ISAAC_PATH/python.sh" ]; then
    echo -e "${RED}Error: python.sh not found at: $ISAAC_PATH/python.sh${NC}"
    echo "The specified path does not appear to be a valid Isaac Sim installation"
    exit 1
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}VLA Local Startup (Isaac Sim)${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "  Demo:         ${GREEN}$DEMO${NC}"
echo -e "  Isaac Sim:    ${GREEN}$ISAAC_PATH${NC}"
if [ "$RECORD" = true ]; then
    echo -e "  Recording:    ${GREEN}Enabled${NC}"
fi
if [ -n "$SERVER_URL" ]; then
    echo -e "  VLA Server:   ${GREEN}$SERVER_URL${NC}"
fi
if [ "$HEADLESS" = true ]; then
    echo -e "  Headless:     ${GREEN}Yes${NC}"
fi
echo ""

# Build the command
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEMO_SCRIPT="$PROJECT_DIR/demos/demo_${DEMO}_control.py"

if [ "$DEMO" = "grasp" ]; then
    DEMO_SCRIPT="$PROJECT_DIR/demos/demo_vla_grasp.py"
fi

if [ ! -f "$DEMO_SCRIPT" ]; then
    echo -e "${RED}Error: Demo script not found: $DEMO_SCRIPT${NC}"
    exit 1
fi

# Build python.sh command
CMD="$ISAAC_PATH/python.sh $DEMO_SCRIPT"

# Add arguments
if [ -n "$SERVER_URL" ]; then
    CMD="$CMD --server $SERVER_URL"
fi

if [ "$RECORD" = true ]; then
    VIDEO_NAME="${DEMO}_demo_$(date +%Y%m%d_%H%M%S).mp4"
    CMD="$CMD --record $VIDEO_NAME"
    echo -e "${BLUE}[INFO] Video will be saved to: output/$VIDEO_NAME${NC}"
fi

if [ "$DEMO" = "grasp" ]; then
    CMD="$CMD --episodes $EPISODES"
fi

if [ "$HEADLESS" = true ]; then
    CMD="$CMD --headless"
fi

echo ""
echo -e "${GREEN}Starting demo with command:${NC}"
echo "  $CMD"
echo ""

# Run the demo
eval $CMD

# Show output location if recording
if [ "$RECORD" = true ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Demo Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "Video saved to: ${GREEN}$PROJECT_DIR/output/$VIDEO_NAME${NC}"
fi
