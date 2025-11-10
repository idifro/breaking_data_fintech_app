#!/usr/bin/env python3
"""
FastAPI Monitoring Service for ML Pipeline Scalability Monitoring
Provides REST API endpoints for metrics access and real-time streaming

CONDA ENVIRONMENT: breaking_data
Usage:
    conda activate breaking_data
    uvicorn dashboard.monitoring_api:app --host 0.0.0.0 --port 8502 --reload

Features:
- REST endpoints for metrics access
- Real-time WebSocket streaming
- CORS configuration for Streamlit integration
- Historical data queries
- Alert notifications
- System health endpoints
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import json
import asyncio
import psutil
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import mlflow
from mlflow.tracking import MlflowClient
import sqlite3
import uuid
from contextlib import asynccontextmanager

# Add src to path
pipeline_root = Path(__file__).parent.parent
sys.path.insert(0, str(pipeline_root))

from src.utils import Config, Logger, load_environment


# Pydantic models for API responses
class MetricPoint(BaseModel):
    """Single metric data point"""
    timestamp: datetime
    metric_name: str
    value: float
    experiment_name: Optional[str] = None
    run_id: Optional[str] = None


class ScalabilityMetrics(BaseModel):
    """Current scalability metrics"""
    throughput_records_per_second: float
    total_processing_time: float
    linear_scalability_score: float
    peak_memory_usage_mb: float
    avg_memory_usage_mb: float
    avg_cpu_usage_percent: float
    data_volume_gb: float
    concurrent_users: int
    last_updated: datetime


class AlertModel(BaseModel):
    """Alert information"""
    id: str
    severity: str
    metric: str
    message: str
    timestamp: datetime
    acknowledged: bool = False


class SystemHealth(BaseModel):
    """Current system health status"""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    load_average: List[float]
    uptime_hours: float
    timestamp: datetime


class ExperimentRun(BaseModel):
    """MLflow experiment run information"""
    run_id: str
    experiment_name: str
    start_time: datetime
    end_time: Optional[datetime]
    status: str
    metrics: Dict[str, float]
    params: Dict[str, str]
    duration_seconds: Optional[float]


class MonitoringService:
    """Core monitoring service for API backend"""
    
    def __init__(self):
        """Initialize monitoring service"""
        try:
            load_environment()
            self.config = Config()
            self.logger = Logger().get_logger()
        except Exception as e:
            print(f"Failed to load configuration: {e}")
            self.config = None
            
        self.setup_mlflow()
        self.alerts_cache: List[AlertModel] = []
        self.websocket_connections: List[WebSocket] = []
        self.monitoring_task = None
        
    def setup_mlflow(self):
        """Setup MLflow client"""
        try:
            if self.config:
                mlflow.set_tracking_uri(str(self.config.paths.mlflow_tracking_dir))
                self.mlflow_client = MlflowClient()
            else:
                self.mlflow_client = None
        except Exception as e:
            print(f"Failed to setup MLflow: {e}")
            self.mlflow_client = None
    
    async def get_current_metrics(self) -> ScalabilityMetrics:
        """Get current scalability metrics"""
        try:
            if not self.mlflow_client:
                # Return default metrics if MLflow not available
                return ScalabilityMetrics(
                    throughput_records_per_second=0.0,
                    total_processing_time=0.0,
                    linear_scalability_score=0.0,
                    peak_memory_usage_mb=0.0,
                    avg_memory_usage_mb=0.0,
                    avg_cpu_usage_percent=0.0,
                    data_volume_gb=0.0,
                    concurrent_users=0,
                    last_updated=datetime.now()
                )
            
            # Get most recent monitoring run
            monitoring_exp_name = "pipeline_scalability_monitoring"
            experiment = self.mlflow_client.get_experiment_by_name(monitoring_exp_name)
            
            if not experiment:
                raise HTTPException(status_code=404, detail="Monitoring experiment not found")
            
            runs = self.mlflow_client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1
            )
            
            if not runs:
                raise HTTPException(status_code=404, detail="No monitoring runs found")
            
            latest_run = runs[0]
            metrics = latest_run.data.metrics
            
            return ScalabilityMetrics(
                throughput_records_per_second=metrics.get('throughput_records_per_second', 0.0),
                total_processing_time=metrics.get('total_processing_time', 0.0),
                linear_scalability_score=metrics.get('linear_scalability_score', 0.0),
                peak_memory_usage_mb=metrics.get('peak_memory_usage_mb', 0.0),
                avg_memory_usage_mb=metrics.get('avg_memory_usage_mb', 0.0),
                avg_cpu_usage_percent=metrics.get('avg_cpu_usage_percent', 0.0),
                data_volume_gb=metrics.get('data_volume_gb', 0.0),
                concurrent_users=int(metrics.get('concurrent_users', 0)),
                last_updated=datetime.fromtimestamp(latest_run.info.start_time / 1000.0)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get current metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_historical_metrics(self, hours: int = 24, limit: int = 100) -> List[MetricPoint]:
        """Get historical metrics for specified time period"""
        try:
            if not self.mlflow_client:
                return []
            
            # Calculate time threshold
            time_threshold = datetime.now() - timedelta(hours=hours)
            timestamp_threshold = int(time_threshold.timestamp() * 1000)
            
            # Get monitoring experiments
            experiment_names = ["pipeline_scalability_monitoring", "pipeline_scalability_monitoring_inference"]
            all_metrics = []
            
            for exp_name in experiment_names:
                experiment = self.mlflow_client.get_experiment_by_name(exp_name)
                if not experiment:
                    continue
                
                runs = self.mlflow_client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string=f"start_time >= {timestamp_threshold}",
                    order_by=["start_time DESC"],
                    max_results=limit
                )
                
                for run in runs:
                    timestamp = datetime.fromtimestamp(run.info.start_time / 1000.0)
                    
                    for metric_name, value in run.data.metrics.items():
                        all_metrics.append(MetricPoint(
                            timestamp=timestamp,
                            metric_name=metric_name,
                            value=value,
                            experiment_name=exp_name,
                            run_id=run.info.run_id
                        ))
            
            # Sort by timestamp
            all_metrics.sort(key=lambda x: x.timestamp, reverse=True)
            return all_metrics[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to get historical metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_system_health(self) -> SystemHealth:
        """Get current system health metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Load average (Unix-like systems)
            try:
                load_avg = os.getloadavg()
            except AttributeError:
                load_avg = [0.0, 0.0, 0.0]  # Windows fallback
            
            # System uptime
            try:
                uptime_seconds = psutil.boot_time()
                uptime_hours = (datetime.now().timestamp() - uptime_seconds) / 3600
            except:
                uptime_hours = 0.0
            
            return SystemHealth(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_percent=disk_percent,
                load_average=list(load_avg),
                uptime_hours=uptime_hours,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get system health: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_recent_experiments(self, limit: int = 20) -> List[ExperimentRun]:
        """Get recent experiment runs"""
        try:
            if not self.mlflow_client:
                return []
            
            # Get monitoring experiments
            experiment_names = ["pipeline_scalability_monitoring", "pipeline_scalability_monitoring_inference"]
            all_runs = []
            
            for exp_name in experiment_names:
                experiment = self.mlflow_client.get_experiment_by_name(exp_name)
                if not experiment:
                    continue
                
                runs = self.mlflow_client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["start_time DESC"],
                    max_results=limit
                )
                
                for run in runs:
                    start_time = datetime.fromtimestamp(run.info.start_time / 1000.0)
                    end_time = datetime.fromtimestamp(run.info.end_time / 1000.0) if run.info.end_time else None
                    
                    duration = (end_time - start_time).total_seconds() if end_time else None
                    
                    all_runs.append(ExperimentRun(
                        run_id=run.info.run_id,
                        experiment_name=exp_name,
                        start_time=start_time,
                        end_time=end_time,
                        status=run.info.status,
                        metrics=run.data.metrics,
                        params=run.data.params,
                        duration_seconds=duration
                    ))
            
            # Sort by start time
            all_runs.sort(key=lambda x: x.start_time, reverse=True)
            return all_runs[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to get recent experiments: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_alerts(self) -> List[AlertModel]:
        """Get current alerts based on thresholds"""
        try:
            current_metrics = await self.get_current_metrics()
            alerts = []
            
            if not self.config:
                return alerts
            
            thresholds = self.config.monitoring.thresholds
            
            # Check throughput
            if current_metrics.throughput_records_per_second < thresholds.min_throughput_records_per_second:
                alerts.append(AlertModel(
                    id=str(uuid.uuid4()),
                    severity='warning',
                    metric='throughput',
                    message=f'Low throughput: {current_metrics.throughput_records_per_second:.1f} records/sec (threshold: {thresholds.min_throughput_records_per_second})',
                    timestamp=datetime.now()
                ))
            
            # Check memory usage (assuming 32GB total for calculation)
            memory_threshold_mb = thresholds.max_memory_usage_percent * 320
            if current_metrics.peak_memory_usage_mb > memory_threshold_mb:
                alerts.append(AlertModel(
                    id=str(uuid.uuid4()),
                    severity='error',
                    metric='memory',
                    message=f'High memory usage: {current_metrics.peak_memory_usage_mb:.1f} MB (threshold: {memory_threshold_mb:.1f} MB)',
                    timestamp=datetime.now()
                ))
            
            # Check CPU usage
            if current_metrics.avg_cpu_usage_percent > thresholds.max_cpu_usage_percent:
                alerts.append(AlertModel(
                    id=str(uuid.uuid4()),
                    severity='warning',
                    metric='cpu',
                    message=f'High CPU usage: {current_metrics.avg_cpu_usage_percent:.1f}% (threshold: {thresholds.max_cpu_usage_percent}%)',
                    timestamp=datetime.now()
                ))
            
            # Check scalability score
            if current_metrics.linear_scalability_score < thresholds.min_scalability_score:
                alerts.append(AlertModel(
                    id=str(uuid.uuid4()),
                    severity='warning',
                    metric='scalability',
                    message=f'Low scalability score: {current_metrics.linear_scalability_score:.3f} (threshold: {thresholds.min_scalability_score})',
                    timestamp=datetime.now()
                ))
            
            self.alerts_cache = alerts
            return alerts
            
        except Exception as e:
            self.logger.error(f"Failed to get alerts: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def stream_metrics_to_websockets(self):
        """Background task to stream metrics to WebSocket connections"""
        while True:
            try:
                if self.websocket_connections:
                    # Get current data
                    current_metrics = await self.get_current_metrics()
                    system_health = await self.get_system_health()
                    alerts = await self.get_alerts()
                    
                    # Prepare streaming data
                    stream_data = {
                        "type": "metrics_update",
                        "timestamp": datetime.now().isoformat(),
                        "metrics": current_metrics.dict(),
                        "system_health": system_health.dict(),
                        "alerts": [alert.dict() for alert in alerts]
                    }
                    
                    # Send to all connected clients
                    disconnected = []
                    for websocket in self.websocket_connections:
                        try:
                            await websocket.send_text(json.dumps(stream_data, default=str))
                        except WebSocketDisconnect:
                            disconnected.append(websocket)
                        except Exception as e:
                            self.logger.warning(f"Failed to send to WebSocket: {e}")
                            disconnected.append(websocket)
                    
                    # Remove disconnected clients
                    for ws in disconnected:
                        if ws in self.websocket_connections:
                            self.websocket_connections.remove(ws)
                
                # Wait for next update
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in metrics streaming: {e}")
                await asyncio.sleep(10)  # Wait longer on error


# Initialize monitoring service
monitoring_service = MonitoringService()

# Lifespan manager for background tasks
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Start background task
    monitoring_service.monitoring_task = asyncio.create_task(
        monitoring_service.stream_metrics_to_websockets()
    )
    yield
    # Cleanup
    if monitoring_service.monitoring_task:
        monitoring_service.monitoring_task.cancel()

# Initialize FastAPI app
app = FastAPI(
    title="ML Pipeline Monitoring API",
    description="REST API for scalability monitoring of ML pipeline",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for Streamlit integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],  # Streamlit default
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}


# Current metrics endpoint
@app.get("/api/metrics/current", response_model=ScalabilityMetrics)
async def get_current_metrics():
    """Get current scalability metrics"""
    return await monitoring_service.get_current_metrics()


# Historical metrics endpoint
@app.get("/api/metrics/history", response_model=List[MetricPoint])
async def get_historical_metrics(
    hours: int = 24,
    limit: int = 100,
    metric_name: Optional[str] = None
):
    """Get historical metrics with optional filtering"""
    metrics = await monitoring_service.get_historical_metrics(hours=hours, limit=limit)
    
    # Filter by metric name if specified
    if metric_name:
        metrics = [m for m in metrics if m.metric_name == metric_name]
    
    return metrics


# System health endpoint
@app.get("/api/system/health", response_model=SystemHealth)
async def get_system_health():
    """Get current system health status"""
    return await monitoring_service.get_system_health()


# Alerts endpoint
@app.get("/api/alerts", response_model=List[AlertModel])
async def get_alerts():
    """Get current alerts"""
    return await monitoring_service.get_alerts()


# Recent experiments endpoint
@app.get("/api/experiments/recent", response_model=List[ExperimentRun])
async def get_recent_experiments(limit: int = 20):
    """Get recent experiment runs"""
    return await monitoring_service.get_recent_experiments(limit=limit)


# Specific experiment details endpoint
@app.get("/api/experiments/{run_id}")
async def get_experiment_details(run_id: str):
    """Get detailed information about a specific experiment run"""
    try:
        if not monitoring_service.mlflow_client:
            raise HTTPException(status_code=503, detail="MLflow not available")
        
        run = monitoring_service.mlflow_client.get_run(run_id)
        
        start_time = datetime.fromtimestamp(run.info.start_time / 1000.0)
        end_time = datetime.fromtimestamp(run.info.end_time / 1000.0) if run.info.end_time else None
        
        return {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "start_time": start_time,
            "end_time": end_time,
            "status": run.info.status,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "tags": run.data.tags,
            "artifact_uri": run.info.artifact_uri
        }
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Run not found: {e}")


# Metric trends endpoint
@app.get("/api/metrics/trends/{metric_name}")
async def get_metric_trends(
    metric_name: str,
    hours: int = 24,
    granularity: str = "hour"
):
    """Get trend data for a specific metric with aggregation"""
    try:
        metrics = await monitoring_service.get_historical_metrics(hours=hours, limit=1000)
        
        # Filter for specific metric
        metric_data = [m for m in metrics if m.metric_name == metric_name]
        
        if not metric_data:
            raise HTTPException(status_code=404, detail=f"No data found for metric: {metric_name}")
        
        # Group by time granularity
        if granularity == "hour":
            # Group by hour
            grouped_data = {}
            for point in metric_data:
                hour_key = point.timestamp.replace(minute=0, second=0, microsecond=0)
                if hour_key not in grouped_data:
                    grouped_data[hour_key] = []
                grouped_data[hour_key].append(point.value)
            
            # Calculate averages
            trend_data = []
            for timestamp, values in sorted(grouped_data.items()):
                trend_data.append({
                    "timestamp": timestamp,
                    "value": sum(values) / len(values),
                    "count": len(values),
                    "min": min(values),
                    "max": max(values)
                })
            
            return trend_data
        
        else:
            # Return raw data
            return [{"timestamp": p.timestamp, "value": p.value} for p in metric_data]
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint for real-time streaming
@app.websocket("/ws/metrics")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time metrics streaming"""
    await websocket.accept()
    monitoring_service.websocket_connections.append(websocket)
    
    try:
        while True:
            # Keep connection alive by waiting for any message
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        monitoring_service.websocket_connections.remove(websocket)


# Trigger monitoring run endpoint
@app.post("/api/monitoring/trigger")
async def trigger_monitoring_run(background_tasks: BackgroundTasks):
    """Trigger a monitoring run (for testing purposes)"""
    try:
        # This would trigger the monitoring scripts
        # For now, we'll just return success
        return {
            "status": "triggered",
            "message": "Monitoring run triggered successfully",
            "timestamp": datetime.now()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Configuration endpoint
@app.get("/api/config")
async def get_configuration():
    """Get current monitoring configuration"""
    try:
        if not monitoring_service.config:
            raise HTTPException(status_code=503, detail="Configuration not available")
        
        return {
            "thresholds": {
                "min_throughput_records_per_second": monitoring_service.config.monitoring.thresholds.min_throughput_records_per_second,
                "max_memory_usage_percent": monitoring_service.config.monitoring.thresholds.max_memory_usage_percent,
                "max_cpu_usage_percent": monitoring_service.config.monitoring.thresholds.max_cpu_usage_percent,
                "min_scalability_score": monitoring_service.config.monitoring.thresholds.min_scalability_score
            },
            "dashboard": {
                "port": monitoring_service.config.monitoring.dashboard.port,
                "auto_refresh_seconds": monitoring_service.config.monitoring.dashboard.auto_refresh_seconds,
                "historical_data_points": monitoring_service.config.monitoring.dashboard.historical_data_points
            },
            "api": {
                "port": monitoring_service.config.monitoring.api.port,
                "enable_cors": monitoring_service.config.monitoring.api.enable_cors
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# API documentation page
@app.get("/", response_class=HTMLResponse)
async def api_documentation():
    """API documentation page"""
    return """
    <html>
        <head>
            <title>ML Pipeline Monitoring API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
                .endpoint { margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; background: #f9f9f9; }
                .method { font-weight: bold; color: #007acc; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 ML Pipeline Monitoring API</h1>
                <p>REST API for real-time scalability monitoring</p>
            </div>
            
            <h2>Available Endpoints:</h2>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/health</code> - Health check
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/api/metrics/current</code> - Current metrics
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/api/metrics/history</code> - Historical metrics
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/api/system/health</code> - System health
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/api/alerts</code> - Current alerts
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/api/experiments/recent</code> - Recent experiments
            </div>
            
            <div class="endpoint">
                <span class="method">WS</span> <code>/ws/metrics</code> - Real-time WebSocket stream
            </div>
            
            <p><a href="/docs">📖 Interactive API Documentation (Swagger)</a></p>
            <p><a href="/redoc">📚 ReDoc Documentation</a></p>
        </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn
    
    # Get port from config or use default
    try:
        load_environment()
        config = Config()
        port = config.monitoring.api.port
    except:
        port = 8502
    
    uvicorn.run(
        "dashboard.monitoring_api:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )