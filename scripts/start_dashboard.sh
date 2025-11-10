#!/bin/bash
"""
MLflow Monitoring Dashboard Startup Script
Starts both FastAPI backend and Streamlit frontend
"""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [[ ! -f "config/config.yaml" ]]; then
    echo -e "${RED}❌ Error: Please run this script from the spark_ml_pipeline directory${NC}"
    echo -e "${YELLOW}💡 Usage: cd spark_ml_pipeline && ./scripts/start_dashboard.sh${NC}"
    exit 1
fi

# Check if conda environment is active
if [[ "$CONDA_DEFAULT_ENV" != "breaking_data" ]]; then
    echo -e "${RED}❌ Error: Please activate the breaking_data conda environment first${NC}"
    echo -e "${YELLOW}💡 Run: conda activate breaking_data${NC}"
    exit 1
fi

echo -e "${BLUE}🚀 Starting MLflow Monitoring Dashboard...${NC}"

# Function to cleanup background processes on exit
cleanup() {
    echo -e "\n${YELLOW}🛑 Shutting down services...${NC}"
    
    # Kill FastAPI process
    if [[ ! -z "$FASTAPI_PID" ]]; then
        kill $FASTAPI_PID 2>/dev/null
        echo -e "${GREEN}✅ FastAPI backend stopped${NC}"
    fi
    
    # Kill Streamlit process
    if [[ ! -z "$STREAMLIT_PID" ]]; then
        kill $STREAMLIT_PID 2>/dev/null
        echo -e "${GREEN}✅ Streamlit frontend stopped${NC}"
    fi
    
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM EXIT

# Start FastAPI backend
echo -e "${BLUE}🔧 Starting FastAPI backend on port 8502...${NC}"
uvicorn dashboard.monitoring_api:app --host 0.0.0.0 --port 8502 --reload &
FASTAPI_PID=$!

# Wait a moment for FastAPI to start
sleep 3

# Check if FastAPI is running
if ! curl -s http://localhost:8502/health > /dev/null; then
    echo -e "${RED}❌ Failed to start FastAPI backend${NC}"
    exit 1
fi

echo -e "${GREEN}✅ FastAPI backend started successfully${NC}"

# Start Streamlit frontend  
echo -e "${BLUE}🎨 Starting Streamlit frontend on port 8501...${NC}"
streamlit run dashboard/monitoring_dashboard.py --server.port 8501 --server.headless true &
STREAMLIT_PID=$!

# Wait a moment for Streamlit to start
sleep 5

echo -e "${GREEN}✅ Streamlit frontend started successfully${NC}"
echo -e "\n${BLUE}🌐 Dashboard URLs:${NC}"
echo -e "${GREEN}  📊 Streamlit Dashboard: http://localhost:8501${NC}"
echo -e "${GREEN}  🔧 FastAPI Backend: http://localhost:8502${NC}" 
echo -e "${GREEN}  📚 API Documentation: http://localhost:8502/docs${NC}"

echo -e "\n${YELLOW}🔄 Services are running. Press Ctrl+C to stop...${NC}"

# Wait for background processes
wait