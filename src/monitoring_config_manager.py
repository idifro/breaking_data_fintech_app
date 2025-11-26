#!/usr/bin/env python3
"""
Monitoring Configuration Manager
Handles automatic updating of MLflow monitoring configuration in config.yaml

Features:
- Auto-update config after monitoring runs
- Track latest run metadata
- Generate MLflow URLs for dashboard integration
"""

import yaml
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient

from src.utils import Logger


class MonitoringConfigManager:
    """Manages monitoring configuration updates"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize config manager
        
        Args:
            config_path: Path to config.yaml file
        """
        self.logger = Logger().get_logger()
        
        if config_path is None:
            self.config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        else:
            self.config_path = config_path
            
        self.mlflow_tracking_uri = None
        self._load_config()
        
    def _load_config(self):
        """Load current configuration"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                
            # Set MLflow tracking URI if available
            if 'paths' in self.config and 'mlflow_tracking_dir' in self.config['paths']:
                self.mlflow_tracking_uri = str(Path(self.config['paths']['mlflow_tracking_dir']))
            else:
                self.mlflow_tracking_uri = "mlflow_tracking"
                
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            self.config = {}
            
    def _save_config(self):
        """Save updated configuration"""
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            self.logger.info(f"✅ Config updated: {self.config_path}")
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
            
    def update_training_monitoring_config(self, run_id: str, experiment_name: str):
        """
        Update training monitoring configuration
        
        Args:
            run_id: MLflow run ID
            experiment_name: MLflow experiment name
        """
        self._update_monitoring_config("training", run_id, experiment_name)
        
    def update_inference_monitoring_config(self, run_id: str, experiment_name: str):
        """
        Update inference monitoring configuration
        
        Args:
            run_id: MLflow run ID
            experiment_name: MLflow experiment name
        """
        self._update_monitoring_config("inference", run_id, experiment_name)
        
    def update_scalability_monitoring_config(self, monitor_type: str, run_id: str, experiment_name: str):
        """
        Update scalability monitoring configuration separately from model runs
        
        Args:
            monitor_type: "training" or "inference"  
            run_id: MLflow run ID
            experiment_name: MLflow experiment name
        """
        try:
            # Set MLflow tracking URI
            mlflow.set_tracking_uri(f"file://{Path.cwd() / self.mlflow_tracking_uri}")
            client = MlflowClient()
            
            # Get run details
            run = client.get_run(run_id)
            
            # Generate URLs
            base_url = f"file://{Path.cwd() / self.mlflow_tracking_uri}"
            metrics_url = f"{base_url}/#/experiments/{run.info.experiment_id}/runs/{run_id}"
            artifacts_url = f"{base_url}/#/experiments/{run.info.experiment_id}/runs/{run_id}/artifacts"
            
            # Ensure monitoring section exists
            if 'mlflow' not in self.config:
                self.config['mlflow'] = {}
            if 'monitoring' not in self.config['mlflow']:
                self.config['mlflow']['monitoring'] = {}
            if monitor_type not in self.config['mlflow']['monitoring']:
                self.config['mlflow']['monitoring'][monitor_type] = {}
            
            # Create scalability-specific entry
            scalability_key = f"{monitor_type}_scalability"
            if scalability_key not in self.config['mlflow']['monitoring']:
                self.config['mlflow']['monitoring'][scalability_key] = {}
                
            # Update scalability monitoring config
            scalability_config = {
                'experiment_name': experiment_name,
                'latest_run_id': run_id,
                'latest_run_name': run.info.run_name,
                'latest_timestamp': datetime.now().isoformat(),
                'metrics_url': metrics_url,
                'artifacts_url': artifacts_url,
                'experiment_id': run.info.experiment_id,
                'run_status': run.info.status,
                'start_time': datetime.fromtimestamp(run.info.start_time / 1000).isoformat() if run.info.start_time else None,
                'end_time': datetime.fromtimestamp(run.info.end_time / 1000).isoformat() if run.info.end_time else None,
                'run_type': 'scalability_monitoring'
            }
            
            self.config['mlflow']['monitoring'][scalability_key].update(scalability_config)
            
            # Save updated config
            self._save_config()
            
            self.logger.info(f"✅ Updated {monitor_type} scalability monitoring config:")
            self.logger.info(f"   Scalability Run ID: {run_id}")
            self.logger.info(f"   Scalability Run Name: {run.info.run_name}")
            self.logger.info(f"   Scalability Experiment: {experiment_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to update {monitor_type} scalability monitoring config: {e}")
    
    def _update_monitoring_config(self, monitor_type: str, run_id: str, experiment_name: str):
        """
        Update monitoring configuration for training or inference
        
        Args:
            monitor_type: "training" or "inference"
            run_id: MLflow run ID
            experiment_name: MLflow experiment name
        """
        try:
            # Set MLflow tracking URI
            mlflow.set_tracking_uri(f"file://{Path.cwd() / self.mlflow_tracking_uri}")
            client = MlflowClient()
            
            # Get run details
            run = client.get_run(run_id)
            
            # Generate URLs
            base_url = f"file://{Path.cwd() / self.mlflow_tracking_uri}"
            metrics_url = f"{base_url}/#/experiments/{run.info.experiment_id}/runs/{run_id}"
            artifacts_url = f"{base_url}/#/experiments/{run.info.experiment_id}/runs/{run_id}/artifacts"
            
            # Ensure monitoring section exists
            if 'mlflow' not in self.config:
                self.config['mlflow'] = {}
            if 'monitoring' not in self.config['mlflow']:
                self.config['mlflow']['monitoring'] = {}
            if monitor_type not in self.config['mlflow']['monitoring']:
                self.config['mlflow']['monitoring'][monitor_type] = {}
                
            # Update monitoring config
            monitoring_config = {
                'experiment_name': experiment_name,
                'latest_run_id': run_id,
                'latest_run_name': run.info.run_name,
                'latest_timestamp': datetime.now().isoformat(),
                'metrics_url': metrics_url,
                'artifacts_url': artifacts_url,
                'experiment_id': run.info.experiment_id,
                'run_status': run.info.status,
                'start_time': datetime.fromtimestamp(run.info.start_time / 1000).isoformat() if run.info.start_time else None,
                'end_time': datetime.fromtimestamp(run.info.end_time / 1000).isoformat() if run.info.end_time else None,
                'run_type': 'model_training' if 'unified_monitoring' in experiment_name else 'scalability_monitoring'
            }
            
            self.config['mlflow']['monitoring'][monitor_type].update(monitoring_config)
            
            # Save updated config
            self._save_config()
            
            self.logger.info(f"✅ Updated {monitor_type} monitoring config:")
            self.logger.info(f"   Run ID: {run_id}")
            self.logger.info(f"   Run Name: {run.info.run_name}")
            self.logger.info(f"   Experiment: {experiment_name}")
            self.logger.info(f"   Metrics URL: {metrics_url}")
            
        except Exception as e:
            self.logger.error(f"Failed to update {monitor_type} monitoring config: {e}")
            
    def get_monitoring_config(self, monitor_type: str) -> Dict[str, Any]:
        """
        Get monitoring configuration for training or inference
        
        Args:
            monitor_type: "training" or "inference"
            
        Returns:
            Dictionary with monitoring configuration
        """
        try:
            return self.config.get('mlflow', {}).get('monitoring', {}).get(monitor_type, {})
        except Exception as e:
            self.logger.error(f"Failed to get {monitor_type} monitoring config: {e}")
            return {}
            
    def get_all_monitoring_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all monitoring configurations
        
        Returns:
            Dictionary with training and inference monitoring configs
        """
        return {
            'training': self.get_monitoring_config('training'),
            'training_scalability': self.get_monitoring_config('training_scalability'),
            'inference': self.get_monitoring_config('inference'), 
            'inference_scalability': self.get_monitoring_config('inference_scalability')
        }
        
    def get_mlflow_tracking_uri(self) -> str:
        """Get MLflow tracking URI"""
        return f"file://{Path.cwd() / self.mlflow_tracking_uri}"
        
    def list_recent_runs(self, monitor_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List recent runs for a monitoring type
        
        Args:
            monitor_type: "training" or "inference"
            limit: Maximum number of runs to return
            
        Returns:
            List of run dictionaries
        """
        try:
            experiment_name = self.get_monitoring_config(monitor_type).get('experiment_name')
            if not experiment_name:
                return []
                
            # Set MLflow tracking URI
            mlflow.set_tracking_uri(self.get_mlflow_tracking_uri())
            client = MlflowClient()
            
            # Get experiment
            try:
                experiment = client.get_experiment_by_name(experiment_name)
                if not experiment:
                    return []
            except Exception:
                return []
                
            # List runs
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=limit,
                order_by=["start_time DESC"]
            )
            
            # Format run data
            run_data = []
            for run in runs:
                run_info = {
                    'run_id': run.info.run_id,
                    'run_name': run.info.run_name,
                    'status': run.info.status,
                    'start_time': datetime.fromtimestamp(run.info.start_time / 1000).isoformat() if run.info.start_time else None,
                    'end_time': datetime.fromtimestamp(run.info.end_time / 1000).isoformat() if run.info.end_time else None,
                    'metrics': dict(run.data.metrics),
                    'params': dict(run.data.params),
                    'tags': dict(run.data.tags)
                }
                run_data.append(run_info)
                
            return run_data
            
        except Exception as e:
            self.logger.error(f"Failed to list recent {monitor_type} runs: {e}")
            return []


# Global instance
_config_manager = None

def get_monitoring_config_manager() -> MonitoringConfigManager:
    """Get global monitoring config manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = MonitoringConfigManager()
    return _config_manager