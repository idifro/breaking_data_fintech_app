#!/usr/bin/env python3
"""
Real-Time Scalability Monitoring Dashboard
Built with Streamlit for visualizing ML pipeline performance metrics

CONDA ENVIRONMENT: breaking_data
Usage: 
    conda activate breaking_data
    streamlit run dashboard/monitoring_dashboard.py --server.port 8501
    
Features:
- Real-time performance metrics visualization
- Historical trend analysis
- Resource utilization monitoring
- Scalability bottleneck identification
- Interactive charts and filters
- Automated refresh capabilities
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import time
import sqlite3
from typing import Dict, List, Optional
import mlflow
from mlflow.tracking import MlflowClient

# Add src to path
pipeline_root = Path(__file__).parent.parent
sys.path.insert(0, str(pipeline_root))

from src.utils import Config, Logger, load_environment


class MonitoringDashboard:
    """Real-time monitoring dashboard for ML pipeline scalability"""
    
    def __init__(self):
        """Initialize dashboard"""
        try:
            load_environment()
            self.config = Config()
            self.logger = Logger().get_logger()
        except Exception as e:
            st.error(f"Failed to load configuration: {e}")
            self.config = None
            
        self.setup_mlflow()
        self.setup_dashboard_config()
    
    def setup_mlflow(self):
        """Setup MLflow client for monitoring data"""
        try:
            if self.config:
                mlflow.set_tracking_uri(str(self.config.paths.mlflow_tracking_dir))
                self.mlflow_client = MlflowClient()
            else:
                self.mlflow_client = None
        except Exception as e:
            st.error(f"Failed to setup MLflow: {e}")
            self.mlflow_client = None
    
    def setup_dashboard_config(self):
        """Setup dashboard configuration"""
        if self.config and hasattr(self.config, 'monitoring'):
            self.auto_refresh = self.config.monitoring.dashboard.auto_refresh_seconds
            self.historical_points = self.config.monitoring.dashboard.historical_data_points
        else:
            self.auto_refresh = 5
            self.historical_points = 100
    
    def load_monitoring_data(self) -> Dict:
        """Load monitoring data from MLflow and file system"""
        monitoring_data = {
            'experiments': [],
            'recent_runs': [],
            'metrics_history': [],
            'current_metrics': {},
            'alerts': []
        }
        
        try:
            if not self.mlflow_client:
                return monitoring_data
            
            # Get monitoring experiments
            monitoring_exp_name = "pipeline_scalability_monitoring"
            inference_exp_name = "pipeline_scalability_monitoring_inference"
            
            for exp_name in [monitoring_exp_name, inference_exp_name]:
                try:
                    experiment = self.mlflow_client.get_experiment_by_name(exp_name)
                    if experiment:
                        monitoring_data['experiments'].append({
                            'name': exp_name,
                            'id': experiment.experiment_id,
                            'lifecycle_stage': experiment.lifecycle_stage
                        })
                        
                        # Get recent runs
                        runs = self.mlflow_client.search_runs(
                            experiment_ids=[experiment.experiment_id],
                            order_by=["start_time DESC"],
                            max_results=self.historical_points
                        )
                        
                        for run in runs:
                            run_data = {
                                'experiment_name': exp_name,
                                'run_id': run.info.run_id,
                                'start_time': run.info.start_time,
                                'end_time': run.info.end_time,
                                'status': run.info.status,
                                'metrics': run.data.metrics,
                                'params': run.data.params,
                                'tags': run.data.tags
                            }
                            monitoring_data['recent_runs'].append(run_data)
                            
                except Exception as e:
                    st.warning(f"Could not load experiment {exp_name}: {e}")
            
            # Sort runs by start time
            monitoring_data['recent_runs'] = sorted(
                monitoring_data['recent_runs'], 
                key=lambda x: x['start_time'], 
                reverse=True
            )
            
            # Extract current metrics from most recent run
            if monitoring_data['recent_runs']:
                latest_run = monitoring_data['recent_runs'][0]
                monitoring_data['current_metrics'] = latest_run['metrics']
            
            # Load metrics history for trending
            monitoring_data['metrics_history'] = self._extract_metrics_history(monitoring_data['recent_runs'])
            
            # Generate alerts based on thresholds
            monitoring_data['alerts'] = self._generate_alerts(monitoring_data['current_metrics'])
            
        except Exception as e:
            st.error(f"Failed to load monitoring data: {e}")
        
        return monitoring_data
    
    def _extract_metrics_history(self, runs: List[Dict]) -> List[Dict]:
        """Extract metrics history for trend analysis"""
        metrics_history = []
        
        for run in runs[:self.historical_points]:  # Limit to configured points
            if run['start_time']:
                timestamp = datetime.fromtimestamp(run['start_time'] / 1000.0)
                
                history_point = {
                    'timestamp': timestamp,
                    'experiment': run['experiment_name'],
                    'run_id': run['run_id']
                }
                
                # Add all metrics
                history_point.update(run['metrics'])
                metrics_history.append(history_point)
        
        return sorted(metrics_history, key=lambda x: x['timestamp'])
    
    def _generate_alerts(self, current_metrics: Dict) -> List[Dict]:
        """Generate alerts based on threshold violations"""
        alerts = []
        
        if not current_metrics or not self.config:
            return alerts
            
        try:
            thresholds = self.config.monitoring.thresholds
            
            # Check throughput
            throughput = current_metrics.get('throughput_records_per_second', 0)
            if throughput < thresholds.min_throughput_records_per_second:
                alerts.append({
                    'severity': 'warning',
                    'metric': 'throughput',
                    'message': f'Low throughput: {throughput:.1f} records/sec (threshold: {thresholds.min_throughput_records_per_second})',
                    'timestamp': datetime.now()
                })
            
            # Check memory usage
            memory_usage = current_metrics.get('peak_memory_usage_mb', 0)
            if memory_usage > thresholds.max_memory_usage_percent * 320:  # Assume 32GB total
                alerts.append({
                    'severity': 'error',
                    'metric': 'memory',
                    'message': f'High memory usage: {memory_usage:.1f} MB',
                    'timestamp': datetime.now()
                })
            
            # Check CPU usage
            cpu_usage = current_metrics.get('avg_cpu_usage_percent', 0)
            if cpu_usage > thresholds.max_cpu_usage_percent:
                alerts.append({
                    'severity': 'warning',
                    'metric': 'cpu',
                    'message': f'High CPU usage: {cpu_usage:.1f}%',
                    'timestamp': datetime.now()
                })
            
            # Check scalability score
            scalability_score = current_metrics.get('linear_scalability_score', 1.0)
            if scalability_score < thresholds.min_scalability_score:
                alerts.append({
                    'severity': 'warning',
                    'metric': 'scalability',
                    'message': f'Low scalability score: {scalability_score:.3f}',
                    'timestamp': datetime.now()
                })
                
        except Exception as e:
            st.warning(f"Could not generate alerts: {e}")
        
        return alerts
    
    def create_overview_metrics(self, monitoring_data: Dict):
        """Create overview metrics cards"""
        current_metrics = monitoring_data.get('current_metrics', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            throughput = current_metrics.get('throughput_records_per_second', 0)
            st.metric(
                label="Throughput",
                value=f"{throughput:.1f} rec/sec",
                delta=None
            )
        
        with col2:
            scalability_score = current_metrics.get('linear_scalability_score', 0)
            st.metric(
                label="Scalability Score", 
                value=f"{scalability_score:.3f}",
                delta=None
            )
        
        with col3:
            memory_usage = current_metrics.get('peak_memory_usage_mb', 0)
            st.metric(
                label="Peak Memory",
                value=f"{memory_usage:.0f} MB",
                delta=None
            )
        
        with col4:
            processing_time = current_metrics.get('total_processing_time', 0)
            st.metric(
                label="Processing Time",
                value=f"{processing_time:.1f}s",
                delta=None
            )
    
    def create_alerts_section(self, monitoring_data: Dict):
        """Create alerts section"""
        alerts = monitoring_data.get('alerts', [])
        
        if alerts:
            st.subheader("🚨 Active Alerts")
            
            for alert in alerts:
                severity_colors = {
                    'error': 'red',
                    'warning': 'orange', 
                    'info': 'blue'
                }
                
                severity_icons = {
                    'error': '🔴',
                    'warning': '🟡',
                    'info': 'ℹ️'
                }
                
                color = severity_colors.get(alert['severity'], 'gray')
                icon = severity_icons.get(alert['severity'], 'ℹ️')
                
                st.markdown(f"""
                <div style="background-color: {color}20; padding: 10px; border-radius: 5px; margin: 5px 0; border-left: 4px solid {color}">
                    {icon} <strong>{alert['metric'].upper()}</strong>: {alert['message']}
                    <br><small>Time: {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No active alerts")
    
    def create_performance_trends(self, monitoring_data: Dict):
        """Create performance trend charts"""
        metrics_history = monitoring_data.get('metrics_history', [])
        
        if not metrics_history:
            st.warning("No historical data available")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(metrics_history)
        
        if df.empty:
            st.warning("No metrics data to display")
            return
        
        # Throughput trend
        st.subheader("📈 Performance Trends")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'throughput_records_per_second' in df.columns:
                fig_throughput = px.line(
                    df, 
                    x='timestamp', 
                    y='throughput_records_per_second',
                    color='experiment',
                    title="Throughput Over Time",
                    labels={'throughput_records_per_second': 'Records/Second', 'timestamp': 'Time'}
                )
                fig_throughput.update_layout(height=400)
                st.plotly_chart(fig_throughput, use_container_width=True)
            else:
                st.info("No throughput data available")
        
        with col2:
            if 'linear_scalability_score' in df.columns:
                fig_scalability = px.line(
                    df,
                    x='timestamp',
                    y='linear_scalability_score',
                    color='experiment', 
                    title="Scalability Score Over Time",
                    labels={'linear_scalability_score': 'Scalability Score', 'timestamp': 'Time'}
                )
                fig_scalability.update_layout(height=400)
                st.plotly_chart(fig_scalability, use_container_width=True)
            else:
                st.info("No scalability data available")
    
    def create_resource_utilization(self, monitoring_data: Dict):
        """Create resource utilization charts"""
        metrics_history = monitoring_data.get('metrics_history', [])
        
        if not metrics_history:
            st.warning("No resource utilization data available")
            return
        
        df = pd.DataFrame(metrics_history)
        
        if df.empty:
            return
        
        st.subheader("💾 Resource Utilization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Memory usage
            memory_cols = [col for col in df.columns if 'memory' in col.lower() and 'mb' in col.lower()]
            if memory_cols:
                fig_memory = go.Figure()
                
                for col in memory_cols:
                    fig_memory.add_trace(go.Scatter(
                        x=df['timestamp'],
                        y=df[col],
                        mode='lines',
                        name=col.replace('_', ' ').title(),
                        fill='tonexty' if col != memory_cols[0] else None
                    ))
                
                fig_memory.update_layout(
                    title="Memory Usage Over Time",
                    xaxis_title="Time",
                    yaxis_title="Memory (MB)",
                    height=400
                )
                st.plotly_chart(fig_memory, use_container_width=True)
            else:
                st.info("No memory data available")
        
        with col2:
            # CPU usage
            cpu_cols = [col for col in df.columns if 'cpu' in col.lower() and 'percent' in col.lower()]
            if cpu_cols:
                fig_cpu = px.line(
                    df,
                    x='timestamp',
                    y=cpu_cols,
                    title="CPU Usage Over Time",
                    labels={'value': 'CPU Usage (%)', 'timestamp': 'Time'}
                )
                fig_cpu.update_layout(height=400)
                st.plotly_chart(fig_cpu, use_container_width=True)
            else:
                st.info("No CPU data available")
    
    def create_experiment_comparison(self, monitoring_data: Dict):
        """Create experiment comparison section"""
        recent_runs = monitoring_data.get('recent_runs', [])
        
        if len(recent_runs) < 2:
            st.info("Need at least 2 runs for comparison")
            return
        
        st.subheader("🔬 Experiment Comparison")
        
        # Select runs to compare
        run_options = [(f"{run['experiment_name']} - {datetime.fromtimestamp(run['start_time']/1000).strftime('%Y-%m-%d %H:%M')}", 
                       run['run_id']) for run in recent_runs[:10]]
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_run1 = st.selectbox("Select first run:", run_options, key="run1")
        
        with col2:
            selected_run2 = st.selectbox("Select second run:", run_options, key="run2", index=1 if len(run_options) > 1 else 0)
        
        if selected_run1 and selected_run2 and selected_run1[1] != selected_run2[1]:
            # Find selected runs
            run1_data = next((run for run in recent_runs if run['run_id'] == selected_run1[1]), None)
            run2_data = next((run for run in recent_runs if run['run_id'] == selected_run2[1]), None)
            
            if run1_data and run2_data:
                self._display_run_comparison(run1_data, run2_data)
    
    def _display_run_comparison(self, run1: Dict, run2: Dict):
        """Display comparison between two runs"""
        comparison_metrics = [
            'throughput_records_per_second',
            'total_processing_time', 
            'linear_scalability_score',
            'peak_memory_usage_mb',
            'avg_cpu_usage_percent'
        ]
        
        comparison_data = []
        for metric in comparison_metrics:
            val1 = run1['metrics'].get(metric, 0)
            val2 = run2['metrics'].get(metric, 0)
            
            # Calculate improvement
            if val1 != 0:
                improvement = ((val2 - val1) / val1) * 100
            else:
                improvement = 0
            
            comparison_data.append({
                'Metric': metric.replace('_', ' ').title(),
                'Run 1': f"{val1:.3f}",
                'Run 2': f"{val2:.3f}", 
                'Improvement %': f"{improvement:+.1f}%"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)
        
        # Create improvement chart
        fig_improvement = px.bar(
            df_comparison,
            x='Metric',
            y=[float(x.replace('%', '')) for x in df_comparison['Improvement %']],
            title="Performance Improvement",
            labels={'y': 'Improvement %'},
            color=[float(x.replace('%', '')) for x in df_comparison['Improvement %']],
            color_continuous_scale='RdYlGn'
        )
        fig_improvement.update_layout(height=400)
        st.plotly_chart(fig_improvement, use_container_width=True)
    
    def create_recent_runs_table(self, monitoring_data: Dict):
        """Create recent runs table"""
        recent_runs = monitoring_data.get('recent_runs', [])
        
        if not recent_runs:
            st.info("No recent runs available")
            return
        
        st.subheader("📋 Recent Runs")
        
        # Prepare table data
        table_data = []
        for run in recent_runs[:20]:  # Show last 20 runs
            start_time = datetime.fromtimestamp(run['start_time'] / 1000.0) if run['start_time'] else None
            end_time = datetime.fromtimestamp(run['end_time'] / 1000.0) if run['end_time'] else None
            
            duration = (end_time - start_time).total_seconds() if start_time and end_time else 0
            
            table_data.append({
                'Experiment': run['experiment_name'].replace('pipeline_scalability_monitoring_', '').title(),
                'Start Time': start_time.strftime('%Y-%m-%d %H:%M:%S') if start_time else 'N/A',
                'Duration (s)': f"{duration:.1f}",
                'Status': run['status'],
                'Throughput': f"{run['metrics'].get('throughput_records_per_second', 0):.1f}",
                'Scalability': f"{run['metrics'].get('linear_scalability_score', 0):.3f}",
                'Memory (MB)': f"{run['metrics'].get('peak_memory_usage_mb', 0):.0f}",
                'Run ID': run['run_id'][:8] + '...'
            })
        
        df_runs = pd.DataFrame(table_data)
        
        if not df_runs.empty:
            # Add filtering
            col1, col2 = st.columns(2)
            
            with col1:
                experiment_filter = st.multiselect(
                    "Filter by experiment:",
                    options=df_runs['Experiment'].unique(),
                    default=df_runs['Experiment'].unique()
                )
            
            with col2:
                status_filter = st.multiselect(
                    "Filter by status:",
                    options=df_runs['Status'].unique(),
                    default=df_runs['Status'].unique()
                )
            
            # Apply filters
            filtered_df = df_runs[
                (df_runs['Experiment'].isin(experiment_filter)) &
                (df_runs['Status'].isin(status_filter))
            ]
            
            st.dataframe(filtered_df, use_container_width=True)
        else:
            st.info("No runs data available")
    
    def create_live_metrics(self):
        """Create live metrics section that updates in real-time"""
        st.subheader("⚡ Live System Metrics")
        
        # Get current system metrics
        try:
            import psutil
            
            # Memory
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = (memory.total - memory.available) / (1024**3)
            
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Disk
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("System Memory", f"{memory_percent:.1f}%", f"{memory_used_gb:.1f} GB")
                
                # Memory gauge
                fig_memory = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = memory_percent,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Memory Usage %"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig_memory.update_layout(height=200)
                st.plotly_chart(fig_memory, use_container_width=True)
            
            with col2:
                st.metric("System CPU", f"{cpu_percent:.1f}%", "")
                
                # CPU gauge
                fig_cpu = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = cpu_percent,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "CPU Usage %"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkgreen"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "red"}
                        ]
                    }
                ))
                fig_cpu.update_layout(height=200)
                st.plotly_chart(fig_cpu, use_container_width=True)
            
            with col3:
                st.metric("System Disk", f"{disk_percent:.1f}%", "")
                
                # Disk gauge
                fig_disk = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = disk_percent,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Disk Usage %"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkorange"},
                        'steps': [
                            {'range': [0, 70], 'color': "lightgray"},
                            {'range': [70, 90], 'color': "yellow"},
                            {'range': [90, 100], 'color': "red"}
                        ]
                    }
                ))
                fig_disk.update_layout(height=200)
                st.plotly_chart(fig_disk, use_container_width=True)
                
        except Exception as e:
            st.error(f"Could not load system metrics: {e}")
    
    def run_dashboard(self):
        """Main dashboard runner"""
        st.set_page_config(
            page_title="ML Pipeline Monitoring",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("📊 ML Pipeline Scalability Monitoring Dashboard")
        st.markdown("Real-time monitoring for Data Engineering at Scale project")
        
        # Sidebar controls
        with st.sidebar:
            st.header("⚙️ Dashboard Controls")
            
            # Auto-refresh toggle
            auto_refresh_enabled = st.checkbox("Auto-refresh", value=True)
            
            if auto_refresh_enabled:
                refresh_interval = st.slider("Refresh interval (seconds)", 1, 30, self.auto_refresh)
                st.markdown(f"🔄 Auto-refreshing every {refresh_interval} seconds")
            
            # Manual refresh
            if st.button("🔄 Refresh Now"):
                st.rerun()
            
            # Data filters
            st.subheader("📊 Data Filters")
            
            time_range = st.selectbox(
                "Time range:",
                ["Last 1 hour", "Last 6 hours", "Last 24 hours", "Last 7 days"],
                index=2
            )
            
            # Display configuration
            st.subheader("📈 Display Options")
            
            show_alerts = st.checkbox("Show alerts", value=True)
            show_trends = st.checkbox("Show performance trends", value=True)
            show_resources = st.checkbox("Show resource utilization", value=True)
            show_comparison = st.checkbox("Show experiment comparison", value=True)
            show_live_metrics = st.checkbox("Show live system metrics", value=True)
            show_runs_table = st.checkbox("Show recent runs table", value=True)
        
        # Load monitoring data
        with st.spinner("Loading monitoring data..."):
            monitoring_data = self.load_monitoring_data()
        
        # Display timestamp
        st.markdown(f"**Last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Overview metrics
        self.create_overview_metrics(monitoring_data)
        
        # Alerts section
        if show_alerts:
            self.create_alerts_section(monitoring_data)
        
        # Live system metrics
        if show_live_metrics:
            self.create_live_metrics()
        
        st.markdown("---")
        
        # Performance trends
        if show_trends:
            self.create_performance_trends(monitoring_data)
        
        st.markdown("---")
        
        # Resource utilization
        if show_resources:
            self.create_resource_utilization(monitoring_data)
        
        st.markdown("---")
        
        # Experiment comparison
        if show_comparison:
            self.create_experiment_comparison(monitoring_data)
        
        st.markdown("---")
        
        # Recent runs table
        if show_runs_table:
            self.create_recent_runs_table(monitoring_data)
        
        # Auto-refresh implementation
        if auto_refresh_enabled:
            time.sleep(refresh_interval)
            st.rerun()


def main():
    """Main dashboard entry point"""
    
    # Add custom CSS
    st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .alert-error {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .alert-warning {
        background-color: #fff8e1;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .alert-info {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize and run dashboard
    dashboard = MonitoringDashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()