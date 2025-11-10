#!/usr/bin/env python3
"""
MLflow Monitoring API for Training and Inference Metrics
Provides REST API endpoints for MLflow-based metrics access and pipeline control

CONDA ENVIRONMENT: breaking_data
Usage:
    conda activate breaking_data
    uvicorn dashboard.monitoring_api:app --host 0.0.0.0 --port 8502 --reload

Features:
- Training and inference metrics from MLflow
- Trigger new training/inference runs with monitoring
- Historical run data and artifact access
- Plot and report file serving
- CORS configuration for Streamlit integration
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Union
import json
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

# Add src to path
pipeline_root = Path(__file__).parent.parent
sys.path.insert(0, str(pipeline_root))

from src.utils import Config, Logger, load_environment
from src.monitoring_config_manager import get_monitoring_config_manager


# Initialize FastAPI app
app = FastAPI(
    title="MLflow Monitoring API",
    description="API for Training and Inference Monitoring Metrics",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for Streamlit integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global configuration
config = None
logger = None
config_manager = None

@app.on_event("startup")
async def startup_event():
    """Initialize configuration and logger"""
    global config, logger, config_manager
    try:
        load_environment()
        config = Config()
        logger = Logger().get_logger()
        config_manager = get_monitoring_config_manager()
        logger.info("🚀 MLflow Monitoring API started")
    except Exception as e:
        print(f"Failed to initialize API: {e}")
        raise


# Pydantic models
class RunTriggerRequest(BaseModel):
    """Request model for triggering training/inference runs"""
    stocks: Optional[List[str]] = None
    enable_monitoring: bool = True
    load_test: bool = False
    additional_params: Optional[Dict[str, Any]] = None


class RunTriggerResponse(BaseModel):
    """Response model for run triggering"""
    status: str
    message: str
    task_id: Optional[str] = None
    estimated_duration: Optional[str] = None


class MetricsResponse(BaseModel):
    """Response model for metrics data"""
    status: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]


class RunInfo(BaseModel):
    """Model for MLflow run information"""
    run_id: str
    run_name: Optional[str]
    status: str
    start_time: Optional[str]
    end_time: Optional[str]
    duration: Optional[float]
    metrics: Dict[str, float]
    params: Dict[str, str]
    tags: Dict[str, str]


# Helper functions
def get_mlflow_client() -> MlflowClient:
    """Get MLflow client with proper tracking URI"""
    mlflow.set_tracking_uri(config_manager.get_mlflow_tracking_uri())
    return MlflowClient()


def format_run_data(run) -> RunInfo:
    """Format MLflow run data for API response"""
    duration = None
    if run.info.start_time and run.info.end_time:
        duration = (run.info.end_time - run.info.start_time) / 1000.0  # Convert to seconds
        
    return RunInfo(
        run_id=run.info.run_id,
        run_name=run.info.run_name,
        status=run.info.status,
        start_time=datetime.fromtimestamp(run.info.start_time / 1000).isoformat() if run.info.start_time else None,
        end_time=datetime.fromtimestamp(run.info.end_time / 1000).isoformat() if run.info.end_time else None,
        duration=duration,
        metrics=dict(run.data.metrics),
        params=dict(run.data.params),
        tags=dict(run.data.tags)
    )


# API Endpoints

@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "MLflow Monitoring API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "training_metrics": "/api/training/metrics",
            "inference_metrics": "/api/inference/metrics",
            "trigger_training": "/api/training/trigger",
            "trigger_inference": "/api/inference/trigger",
            "plots": "/api/plots",
            "reports": "/api/reports"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check MLflow connectivity
        client = get_mlflow_client()
        experiments = client.search_experiments()
        
        # Check config manager
        monitoring_configs = config_manager.get_all_monitoring_configs()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "mlflow_experiments": len(experiments),
            "monitoring_configs": {
                "training": bool(monitoring_configs.get('training')),
                "inference": bool(monitoring_configs.get('inference'))
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


# Training Monitoring Endpoints

@app.get("/api/training/metrics", response_model=MetricsResponse)
async def get_training_metrics(
    limit: int = Query(10, ge=1, le=100, description="Number of recent runs to fetch"),
    run_id: Optional[str] = Query(None, description="Specific run ID to fetch")
):
    """Get training monitoring metrics from MLflow"""
    try:
        if run_id:
            # Get specific run
            client = get_mlflow_client()
            run = client.get_run(run_id)
            runs_data = [format_run_data(run)]
        else:
            # Get recent runs
            runs_data = config_manager.list_recent_runs("training", limit)
            
        training_config = config_manager.get_monitoring_config("training")
        
        response_data = {
            "recent_runs": runs_data,
            "latest_run": runs_data[0] if runs_data else None,
            "experiment_name": training_config.get("experiment_name"),
            "total_runs": len(runs_data)
        }
        
        return MetricsResponse(
            status="success",
            data=response_data,
            metadata={
                "experiment_config": training_config,
                "mlflow_uri": config_manager.get_mlflow_tracking_uri(),
                "fetched_at": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get training metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch training metrics: {str(e)}")


@app.get("/api/training/run/{run_id}/details")
async def get_training_run_details(run_id: str):
    """Get detailed information for a specific training run"""
    try:
        client = get_mlflow_client()
        run = client.get_run(run_id)
        
        # Get artifacts
        artifacts = []
        try:
            artifact_list = client.list_artifacts(run_id)
            artifacts = [{"path": art.path, "is_dir": art.is_dir, "file_size": art.file_size} for art in artifact_list]
        except Exception as e:
            logger.warning(f"Could not fetch artifacts for run {run_id}: {e}")
            
        return {
            "status": "success",
            "run": format_run_data(run),
            "artifacts": artifacts,
            "experiment_id": run.info.experiment_id
        }
        
    except Exception as e:
        logger.error(f"Failed to get training run details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch run details: {str(e)}")


@app.post("/api/training/trigger", response_model=RunTriggerResponse)
async def trigger_training_run(request: RunTriggerRequest, background_tasks: BackgroundTasks):
    """Trigger a new training run with monitoring"""
    try:
        # Prepare command
        cmd = ["python", "scripts/train_model_with_monitoring.py"]
        
        if request.stocks:
            cmd.extend(["--stocks", ",".join(request.stocks)])
            
        if not request.enable_monitoring:
            cmd.append("--no-monitoring")
            
        if request.load_test:
            cmd.append("--load-test")
            
        # Run in background
        task_id = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        def run_training():
            try:
                logger.info(f"🚀 Starting training task {task_id}: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=pipeline_root)
                
                if result.returncode == 0:
                    logger.info(f"✅ Training task {task_id} completed successfully")
                else:
                    logger.error(f"❌ Training task {task_id} failed: {result.stderr}")
                    
            except Exception as e:
                logger.error(f"Training task {task_id} exception: {e}")
                
        background_tasks.add_task(run_training)
        
        return RunTriggerResponse(
            status="started",
            message=f"Training run started with task ID: {task_id}",
            task_id=task_id,
            estimated_duration="10-30 minutes"
        )
        
    except Exception as e:
        logger.error(f"Failed to trigger training run: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")


# Inference Monitoring Endpoints

@app.get("/api/inference/metrics", response_model=MetricsResponse)
async def get_inference_metrics(
    limit: int = Query(10, ge=1, le=100, description="Number of recent runs to fetch"),
    run_id: Optional[str] = Query(None, description="Specific run ID to fetch")
):
    """Get inference monitoring metrics from MLflow"""
    try:
        if run_id:
            # Get specific run
            client = get_mlflow_client()
            run = client.get_run(run_id)
            runs_data = [format_run_data(run)]
        else:
            # Get recent runs
            runs_data = config_manager.list_recent_runs("inference", limit)
            
        inference_config = config_manager.get_monitoring_config("inference")
        
        response_data = {
            "recent_runs": runs_data,
            "latest_run": runs_data[0] if runs_data else None,
            "experiment_name": inference_config.get("experiment_name"),
            "total_runs": len(runs_data)
        }
        
        return MetricsResponse(
            status="success",
            data=response_data,
            metadata={
                "experiment_config": inference_config,
                "mlflow_uri": config_manager.get_mlflow_tracking_uri(),
                "fetched_at": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get inference metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch inference metrics: {str(e)}")


@app.get("/api/inference/run/{run_id}/details")
async def get_inference_run_details(run_id: str):
    """Get detailed information for a specific inference run"""
    try:
        client = get_mlflow_client()
        run = client.get_run(run_id)
        
        # Get artifacts
        artifacts = []
        try:
            artifact_list = client.list_artifacts(run_id)
            artifacts = [{"path": art.path, "is_dir": art.is_dir, "file_size": art.file_size} for art in artifact_list]
        except Exception as e:
            logger.warning(f"Could not fetch artifacts for run {run_id}: {e}")
            
        return {
            "status": "success",
            "run": format_run_data(run),
            "artifacts": artifacts,
            "experiment_id": run.info.experiment_id
        }
        
    except Exception as e:
        logger.error(f"Failed to get inference run details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch run details: {str(e)}")


@app.post("/api/inference/trigger", response_model=RunTriggerResponse)
async def trigger_inference_run(request: RunTriggerRequest, background_tasks: BackgroundTasks):
    """Trigger a new inference run with monitoring"""
    try:
        # Prepare command
        cmd = ["python", "scripts/inference_with_monitoring.py"]
        
        if request.stocks:
            cmd.extend(["--stocks", ",".join(request.stocks)])
            
        if not request.enable_monitoring:
            cmd.append("--no-monitoring")
            
        if request.load_test:
            cmd.append("--load-test")
            
        # Run in background
        task_id = f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        def run_inference():
            try:
                logger.info(f"🚀 Starting inference task {task_id}: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=pipeline_root)
                
                if result.returncode == 0:
                    logger.info(f"✅ Inference task {task_id} completed successfully")
                else:
                    logger.error(f"❌ Inference task {task_id} failed: {result.stderr}")
                    
            except Exception as e:
                logger.error(f"Inference task {task_id} exception: {e}")
                
        background_tasks.add_task(run_inference)
        
        return RunTriggerResponse(
            status="started",
            message=f"Inference run started with task ID: {task_id}",
            task_id=task_id,
            estimated_duration="5-15 minutes"
        )
        
    except Exception as e:
        logger.error(f"Failed to trigger inference run: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start inference: {str(e)}")


# File Serving Endpoints

@app.get("/api/plots")
async def list_plots():
    """List available plot files"""
    try:
        plots_dir = pipeline_root / "plots"
        if not plots_dir.exists():
            return {"plots": [], "message": "No plots directory found"}
            
        plot_files = []
        for plot_file in plots_dir.rglob("*.png"):
            relative_path = plot_file.relative_to(plots_dir)
            plot_files.append({
                "name": plot_file.name,
                "path": str(relative_path),
                "size": plot_file.stat().st_size,
                "modified": datetime.fromtimestamp(plot_file.stat().st_mtime).isoformat()
            })
            
        return {
            "plots": sorted(plot_files, key=lambda x: x["modified"], reverse=True),
            "total_count": len(plot_files)
        }
        
    except Exception as e:
        logger.error(f"Failed to list plots: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list plots: {str(e)}")


@app.get("/api/plots/{plot_path:path}")
async def get_plot(plot_path: str):
    """Serve plot files"""
    try:
        plots_dir = pipeline_root / "plots"
        file_path = plots_dir / plot_path
        
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="Plot file not found")
            
        # Security check - ensure file is within plots directory
        if not str(file_path.resolve()).startswith(str(plots_dir.resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
            
        return FileResponse(
            path=file_path,
            media_type="image/png",
            filename=file_path.name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to serve plot: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to serve plot: {str(e)}")


@app.get("/api/reports")
async def list_reports():
    """List available report files"""
    try:
        reports_dirs = [
            pipeline_root / "results" / "training",
            pipeline_root / "results" / "inference"
        ]
        
        report_files = []
        for reports_dir in reports_dirs:
            if reports_dir.exists():
                report_type = reports_dir.name
                for report_file in reports_dir.glob("*.txt"):
                    report_files.append({
                        "name": report_file.name,
                        "type": report_type,
                        "path": str(report_file.relative_to(pipeline_root)),
                        "size": report_file.stat().st_size,
                        "modified": datetime.fromtimestamp(report_file.stat().st_mtime).isoformat()
                    })
                    
        return {
            "reports": sorted(report_files, key=lambda x: x["modified"], reverse=True),
            "total_count": len(report_files)
        }
        
    except Exception as e:
        logger.error(f"Failed to list reports: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list reports: {str(e)}")


@app.get("/api/reports/{report_path:path}")
async def get_report(report_path: str):
    """Serve report files"""
    try:
        file_path = pipeline_root / report_path
        
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="Report file not found")
            
        # Security check - ensure file is within results directory
        results_dir = pipeline_root / "results"
        if not str(file_path.resolve()).startswith(str(results_dir.resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
            
        return FileResponse(
            path=file_path,
            media_type="text/plain",
            filename=file_path.name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to serve report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to serve report: {str(e)}")


# Configuration Endpoints

@app.get("/api/config/monitoring")
async def get_monitoring_config():
    """Get current monitoring configuration"""
    try:
        return {
            "status": "success",
            "config": config_manager.get_all_monitoring_configs(),
            "mlflow_uri": config_manager.get_mlflow_tracking_uri()
        }
    except Exception as e:
        logger.error(f"Failed to get monitoring config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get config: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8502)