#!/usr/bin/env python3
"""
MLflow Monitoring Dashboard - Streamlit Frontend
Real-time monitoring interface for training and inference metrics via MLflow

CONDA ENVIRONMENT: breaking_data
Usage:
    conda activate breaking_data
    streamlit run dashboard/monitoring_dashboard.py --server.port 8501

Features:
- Training metrics dashboard with MLflow integration
- Inference metrics dashboard with MLflow integration  
- Real-time monitoring data via FastAPI backend
- Trigger new training/inference runs
- Plot visualization from pipeline results
- Report viewing capabilities
"""

import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys

# Add src to path for utilities
pipeline_root = Path(__file__).parent.parent
sys.path.insert(0, str(pipeline_root))

# API Configuration
API_BASE_URL = "http://localhost:8502"


# Page configuration
st.set_page_config(
    page_title="📊 MLflow Monitoring Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .error-metric {
        border-left-color: #dc3545;
    }
    .run-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sidebar-section {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# Utility functions
@st.cache_data(ttl=60)
def fetch_api_data(endpoint: str) -> Dict:
    """Fetch data from API with caching"""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch data from {endpoint}: {str(e)}")
        return {}


def format_duration(seconds: Optional[float]) -> str:
    """Format duration in human-readable format"""
    if not seconds:
        return "N/A"
    
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def format_timestamp(timestamp_str: Optional[str]) -> str:
    """Format timestamp for display"""
    if not timestamp_str:
        return "N/A"
    
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp_str


def get_status_color(status: str) -> str:
    """Get color for run status"""
    status_colors = {
        "FINISHED": "🟢",
        "RUNNING": "🟡", 
        "FAILED": "🔴",
        "KILLED": "🟠",
        "SCHEDULED": "⚪"
    }
    return status_colors.get(status, "⚫")


def trigger_run(run_type: str, params: Dict) -> bool:
    """Trigger a new training or inference run"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/{run_type}/trigger",
            json=params,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        
        st.success(f"✅ {run_type.title()} run started!")
        st.info(f"Task ID: {result.get('task_id')}")
        st.info(f"Estimated duration: {result.get('estimated_duration')}")
        return True
        
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to trigger {run_type} run: {str(e)}")
        return False


# Main application
def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 MLflow Monitoring Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🔧 Dashboard Controls")
        
        # API Health Check
        with st.container():
            st.markdown("### 🏥 System Status")
            health_data = fetch_api_data("/health")
            
            if health_data.get("status") == "healthy":
                st.success("✅ API Connected")
                st.metric("MLflow Experiments", health_data.get("mlflow_experiments", 0))
            else:
                st.error("❌ API Disconnected")
        
        # Refresh Controls
        st.markdown("---")
        auto_refresh = st.checkbox("🔄 Auto Refresh", value=True)
        refresh_interval = st.slider("Refresh Interval (seconds)", 10, 300, 60)
        
        if st.button("🔄 Refresh Now"):
            st.cache_data.clear()
            st.rerun()
        
        # Run Triggers
        st.markdown("---")
        st.markdown("### 🚀 Trigger New Runs")
        
        with st.expander("📊 Training Run", expanded=False):
            train_stocks = st.multiselect(
                "Select Stocks", 
                ["AAPL", "GOOG", "TSLA", "NVDA", "BABA"],
                default=["AAPL", "GOOG"]
            )
            train_monitoring = st.checkbox("Enable Monitoring", value=True, key="train_mon")
            train_load_test = st.checkbox("Load Test Mode", value=False, key="train_load")
            
            if st.button("🚀 Start Training", key="start_train"):
                params = {
                    "stocks": train_stocks,
                    "enable_monitoring": train_monitoring,
                    "load_test": train_load_test
                }
                trigger_run("training", params)
        
        with st.expander("⚡ Inference Run", expanded=False):
            infer_stocks = st.multiselect(
                "Select Stocks", 
                ["AAPL", "GOOG", "TSLA", "NVDA", "BABA"],
                default=["AAPL"],
                key="infer_stocks"
            )
            infer_monitoring = st.checkbox("Enable Monitoring", value=True, key="infer_mon")
            infer_load_test = st.checkbox("Load Test Mode", value=False, key="infer_load")
            
            if st.button("⚡ Start Inference", key="start_infer"):
                params = {
                    "stocks": infer_stocks,
                    "enable_monitoring": infer_monitoring,
                    "load_test": infer_load_test
                }
                trigger_run("inference", params)

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Training Metrics", "⚡ Inference Metrics", "📈 Plots", "📄 Reports"])
    
    with tab1:
        show_training_dashboard()
    
    with tab2:
        show_inference_dashboard()
    
    with tab3:
        show_plots_dashboard()
    
    with tab4:
        show_reports_dashboard()
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


def show_training_dashboard():
    """Display training metrics dashboard"""
    st.markdown("## 📊 Training Monitoring Dashboard")
    
    # Fetch training metrics
    training_data = fetch_api_data("/api/training/metrics?limit=10")
    
    if not training_data or training_data.get("status") != "success":
        st.warning("No training data available")
        return
    
    data = training_data["data"]
    metadata = training_data["metadata"]
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Runs", 
            data.get("total_runs", 0),
            help="Total number of training runs"
        )
    
    with col2:
        latest_run = data.get("latest_run")
        if latest_run:
            status_color = get_status_color(latest_run["status"])
            st.metric(
                "Latest Status", 
                f"{status_color} {latest_run['status']}",
                help="Status of most recent training run"
            )
    
    with col3:
        experiment_name = data.get("experiment_name", "N/A")
        st.metric(
            "Experiment", 
            experiment_name,
            help="Current MLflow experiment"
        )
    
    with col4:
        if latest_run and latest_run.get("duration"):
            duration_str = format_duration(latest_run["duration"])
            st.metric(
                "Last Duration", 
                duration_str,
                help="Duration of most recent run"
            )
    
    # Latest run details
    if latest_run:
        st.markdown("### 🎯 Latest Training Run")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**Run ID:** `{latest_run['run_id']}`")
            st.markdown(f"**Started:** {format_timestamp(latest_run['start_time'])}")
            st.markdown(f"**Finished:** {format_timestamp(latest_run['end_time'])}")
            
        with col2:
            if st.button("🔍 View Details", key="train_details"):
                show_run_details(latest_run["run_id"], "training")
    
        # Metrics visualization
        if latest_run.get("metrics"):
            st.markdown("### 📈 Training Metrics")
            
            metrics_df = pd.DataFrame([
                {"Metric": k, "Value": v} 
                for k, v in latest_run["metrics"].items()
            ])
            
            # Create metrics chart
            if not metrics_df.empty:
                fig = px.bar(
                    metrics_df, 
                    x="Metric", 
                    y="Value",
                    title="Latest Training Run Metrics",
                    color="Value",
                    color_continuous_scale="Blues"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    # Historical runs table
    recent_runs = data.get("recent_runs", [])
    if recent_runs:
        st.markdown("### 📋 Recent Training Runs")
        
        # Convert to DataFrame
        runs_df = pd.DataFrame([
            {
                "Run ID": run["run_id"][:8] + "...",
                "Status": f"{get_status_color(run['status'])} {run['status']}",
                "Start Time": format_timestamp(run["start_time"]),
                "Duration": format_duration(run.get("duration")),
                "Key Metrics": len(run.get("metrics", {}))
            }
            for run in recent_runs[:10]
        ])
        
        st.dataframe(runs_df, use_container_width=True)
    
    # MLflow link
    mlflow_uri = metadata.get("mlflow_uri")
    if mlflow_uri:
        st.markdown(f"🔗 [Open in MLflow]({mlflow_uri})")


def show_inference_dashboard():
    """Display inference metrics dashboard"""
    st.markdown("## ⚡ Inference Monitoring Dashboard")
    
    # Fetch inference metrics
    inference_data = fetch_api_data("/api/inference/metrics?limit=10")
    
    if not inference_data or inference_data.get("status") != "success":
        st.warning("No inference data available")
        return
    
    data = inference_data["data"]
    metadata = inference_data["metadata"]
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Runs", 
            data.get("total_runs", 0),
            help="Total number of inference runs"
        )
    
    with col2:
        latest_run = data.get("latest_run")
        if latest_run:
            status_color = get_status_color(latest_run["status"])
            st.metric(
                "Latest Status", 
                f"{status_color} {latest_run['status']}",
                help="Status of most recent inference run"
            )
    
    with col3:
        experiment_name = data.get("experiment_name", "N/A")
        st.metric(
            "Experiment", 
            experiment_name,
            help="Current MLflow experiment"
        )
    
    with col4:
        if latest_run and latest_run.get("duration"):
            duration_str = format_duration(latest_run["duration"])
            st.metric(
                "Last Duration", 
                duration_str,
                help="Duration of most recent run"
            )
    
    # Latest run details
    if latest_run:
        st.markdown("### 🎯 Latest Inference Run")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**Run ID:** `{latest_run['run_id']}`")
            st.markdown(f"**Started:** {format_timestamp(latest_run['start_time'])}")
            st.markdown(f"**Finished:** {format_timestamp(latest_run['end_time'])}")
            
        with col2:
            if st.button("🔍 View Details", key="infer_details"):
                show_run_details(latest_run["run_id"], "inference")
    
        # Metrics visualization for inference
        if latest_run.get("metrics"):
            st.markdown("### 📊 Inference Performance")
            
            # Create performance overview
            col1, col2, col3 = st.columns(3)
            
            metrics = latest_run["metrics"]
            
            # Latency metrics
            with col1:
                latency_score = metrics.get("efficiency_latency_score", 0)
                st.metric(
                    "Latency Score", 
                    f"{latency_score:.3f}",
                    delta=None,
                    help="Lower is better - inference latency efficiency"
                )
            
            # Throughput metrics  
            with col2:
                throughput_score = metrics.get("efficiency_throughput_score", 0)
                st.metric(
                    "Throughput Score", 
                    f"{throughput_score:.3f}",
                    delta=None,
                    help="Higher is better - prediction throughput"
                )
            
            # Resource efficiency
            with col3:
                resource_score = metrics.get("efficiency_resource_score", 0)
                st.metric(
                    "Resource Score", 
                    f"{resource_score:.3f}",
                    delta=None,
                    help="Resource utilization efficiency"
                )
    
    # Historical runs table
    recent_runs = data.get("recent_runs", [])
    if recent_runs:
        st.markdown("### 📋 Recent Inference Runs")
        
        # Convert to DataFrame
        runs_df = pd.DataFrame([
            {
                "Run ID": run["run_id"][:8] + "...",
                "Status": f"{get_status_color(run['status'])} {run['status']}",
                "Start Time": format_timestamp(run["start_time"]),
                "Duration": format_duration(run.get("duration")),
                "Efficiency": f"{run.get('metrics', {}).get('efficiency_overall_score', 0):.3f}"
            }
            for run in recent_runs[:10]
        ])
        
        st.dataframe(runs_df, use_container_width=True)
    
    # MLflow link
    mlflow_uri = metadata.get("mlflow_uri")
    if mlflow_uri:
        st.markdown(f"🔗 [Open in MLflow]({mlflow_uri})")


def show_run_details(run_id: str, run_type: str):
    """Show detailed information for a specific run"""
    details_data = fetch_api_data(f"/api/{run_type}/run/{run_id}/details")
    
    if not details_data or details_data.get("status") != "success":
        st.error("Failed to fetch run details")
        return
    
    run_info = details_data["run"]
    artifacts = details_data.get("artifacts", [])
    
    # Create a modal-like expander
    with st.expander(f"🔍 Run Details: {run_id}", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Run Information")
            st.json({
                "Run ID": run_info["run_id"],
                "Status": run_info["status"],
                "Start Time": run_info["start_time"],
                "End Time": run_info["end_time"],
                "Duration": f"{run_info.get('duration', 0):.2f}s"
            })
        
        with col2:
            st.markdown("#### 📈 Metrics")
            if run_info.get("metrics"):
                metrics_df = pd.DataFrame([
                    {"Metric": k, "Value": f"{v:.6f}" if isinstance(v, float) else str(v)}
                    for k, v in run_info["metrics"].items()
                ])
                st.dataframe(metrics_df, use_container_width=True)
        
        # Parameters
        if run_info.get("params"):
            st.markdown("#### ⚙️ Parameters")
            st.json(run_info["params"])
        
        # Artifacts
        if artifacts:
            st.markdown("#### 📎 Artifacts")
            artifacts_df = pd.DataFrame(artifacts)
            st.dataframe(artifacts_df, use_container_width=True)


def show_plots_dashboard():
    """Display plots from the pipeline"""
    st.markdown("## 📈 Generated Plots")
    
    plots_data = fetch_api_data("/api/plots")
    
    if not plots_data or not plots_data.get("plots"):
        st.info("No plots available")
        return
    
    plots = plots_data["plots"]
    
    # Plot filtering
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"**{plots_data['total_count']} plots available**")
    
    with col2:
        sort_by = st.selectbox("Sort by", ["Modified", "Name"], key="plot_sort")
    
    # Display plots
    for i, plot in enumerate(plots):
        with st.expander(f"📊 {plot['name']}", expanded=(i < 3)):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                try:
                    # Display image
                    plot_url = f"{API_BASE_URL}/api/plots/{plot['path']}"
                    st.image(plot_url, caption=plot['name'], use_column_width=True)
                except Exception as e:
                    st.error(f"Failed to load plot: {e}")
            
            with col2:
                st.markdown("**Plot Info:**")
                st.markdown(f"Size: {plot['size']:,} bytes")
                st.markdown(f"Modified: {format_timestamp(plot['modified'])}")
                
                if st.button("📥 Download", key=f"download_plot_{i}"):
                    st.markdown(f"[Download]({plot_url})")


def show_reports_dashboard():
    """Display generated reports"""
    st.markdown("## 📄 Generated Reports")
    
    reports_data = fetch_api_data("/api/reports")
    
    if not reports_data or not reports_data.get("reports"):
        st.info("No reports available")
        return
    
    reports = reports_data["reports"]
    
    # Report filtering
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"**{reports_data['total_count']} reports available**")
    
    with col2:
        filter_type = st.selectbox("Filter by Type", ["All", "Training", "Inference"], key="report_filter")
    
    # Filter reports
    filtered_reports = reports
    if filter_type != "All":
        filtered_reports = [r for r in reports if r["type"].lower() == filter_type.lower()]
    
    # Display reports
    for i, report in enumerate(filtered_reports[:10]):
        with st.expander(f"📄 {report['name']} ({report['type']})", expanded=(i < 3)):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                try:
                    # Fetch and display report content
                    report_url = f"{API_BASE_URL}/api/reports/{report['path']}"
                    response = requests.get(report_url)
                    response.raise_for_status()
                    
                    st.code(response.text, language="text")
                    
                except Exception as e:
                    st.error(f"Failed to load report: {e}")
            
            with col2:
                st.markdown("**Report Info:**")
                st.markdown(f"Type: {report['type']}")
                st.markdown(f"Size: {report['size']:,} bytes")
                st.markdown(f"Modified: {format_timestamp(report['modified'])}")
                
                if st.button("📥 Download", key=f"download_report_{i}"):
                    st.markdown(f"[Download]({report_url})")


if __name__ == "__main__":
    main()