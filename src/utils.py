"""
Configuration and utility modules for Spark ML Pipeline
"""

import os
import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import mlflow

# Set up basic logger instead of loguru
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SparkConfig:
    """Spark configuration settings"""
    app_name: str
    master: str
    driver_memory: str
    executor_memory: str
    executor_cores: int
    configs: Dict[str, str]


@dataclass
class ModelConfig:
    """Model configuration settings"""
    algorithm: str
    default_params: Dict[str, Any]
    hyperparameter_ranges: Dict[str, List]


@dataclass
class DataConfig:
    """Data configuration settings"""
    sequence_length: int
    train_split: float
    target_column: str
    sentiment_columns: List[str]
    
    # Stock selection for training and inference
    available_stocks: List[str]
    training_stocks: List[str]
    inference_stocks: List[str]
    
    # Delta table configuration
    stock_tables_prefix: str
    inference_tables_prefix: str


@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    price_lags: List[int]
    volume_lags: List[int]
    sentiment_lags: List[int]
    ma_windows: List[int]
    enable_rsi: bool
    enable_volatility: bool
    enable_interactions: bool
    max_features: int
    correlation_threshold: float


@dataclass
class InferenceConfig:
    """Inference configuration settings"""
    lookback_days: int
    model_name: str
    model_stage: str
    model_version: str
    predict_next_trading_day: bool
    skip_weekends_holidays: bool
    save_predictions: bool
    prediction_column_name: str
    min_required_rows: int
    validate_data_quality: bool
    quality_checks: Dict[str, Any]


@dataclass 
class MonitoringThresholds:
    """Monitoring thresholds configuration"""
    min_throughput_records_per_second: float
    max_processing_time_per_record_ms: float
    min_partition_efficiency: float
    min_scalability_score: float
    max_memory_usage_percent: float
    max_cpu_usage_percent: float


@dataclass
class MonitoringDashboard:
    """Monitoring dashboard configuration"""
    enabled: bool
    port: int
    auto_refresh_seconds: int
    historical_data_points: int


@dataclass
class MonitoringAPI:
    """Monitoring API configuration"""
    enabled: bool
    port: int
    cors_origins: List[str]


@dataclass
class MonitoringLoadTesting:
    """Monitoring load testing configuration"""
    data_volume_multipliers: List[int]
    concurrent_users: List[int]
    benchmark_duration_minutes: int


@dataclass
class MonitoringAlerts:
    """Monitoring alerts configuration"""
    enabled: bool
    email_notifications: bool
    slack_webhook: Optional[str]
    performance_degradation_threshold: float


@dataclass
class MonitoringConfig:
    """Scalability monitoring configuration"""
    enabled: bool
    detailed_metrics: bool
    experiment_name: str
    separate_experiment: bool
    resource_monitoring_interval: float
    memory_threshold_warning_mb: int
    cpu_threshold_warning_percent: int
    thresholds: MonitoringThresholds
    generate_reports: bool
    save_metrics_json: bool
    console_output: bool
    dashboard: MonitoringDashboard
    api: MonitoringAPI
    load_testing: MonitoringLoadTesting
    alerts: MonitoringAlerts


class Config:
    """Main configuration class"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self._config = self._load_config()
        
        # Initialize sub-configurations
        self.data = DataConfig(**self._config['data'])
        self.features = FeatureConfig(**self._config['features'])
        self.model = ModelConfig(**self._config['model'])
        self.spark = SparkConfig(**self._config['spark'])
        self.mlflow_config = self._config['mlflow']
        self.evaluation = self._config['evaluation']
        self.hyperparameter_tuning = self._config['hyperparameter_tuning']
        self.logging = self._config['logging']
        self.visualization = self._config['visualization']
        
        # Initialize path manager
        self.paths = PathManager()
        
        # Initialize inference configuration if available
        if 'inference' in self._config:
            self.inference = InferenceConfig(**self._config['inference'])
        else:
            # Provide default inference configuration
            self.inference = InferenceConfig(
                lookback_days=49,
                model_name="unified_stock_model",
                model_stage="None",
                model_version="latest",
                predict_next_trading_day=True,
                skip_weekends_holidays=True,
                save_predictions=False,
                prediction_column_name="predicted_close",
                min_required_rows=50,
                validate_data_quality=True,
                quality_checks={
                    "check_missing_values": True,
                    "check_data_continuity": True,
                    "max_missing_ratio": 0.1
                }
            )
        
        # Initialize monitoring configuration if available
        if 'monitoring' in self._config:
            monitoring_data = self._config['monitoring']
            self.monitoring = MonitoringConfig(
                enabled=monitoring_data.get('enabled', True),
                detailed_metrics=monitoring_data.get('detailed_metrics', True),
                experiment_name=monitoring_data.get('experiment_name', 'pipeline_scalability_monitoring'),
                separate_experiment=monitoring_data.get('separate_experiment', True),
                resource_monitoring_interval=monitoring_data.get('resource_monitoring_interval', 1.0),
                memory_threshold_warning_mb=monitoring_data.get('memory_threshold_warning_mb', 8192),
                cpu_threshold_warning_percent=monitoring_data.get('cpu_threshold_warning_percent', 85),
                thresholds=MonitoringThresholds(**monitoring_data.get('thresholds', {})),
                generate_reports=monitoring_data.get('generate_reports', True),
                save_metrics_json=monitoring_data.get('save_metrics_json', True),
                console_output=monitoring_data.get('console_output', True),
                dashboard=MonitoringDashboard(**monitoring_data.get('dashboard', {})),
                api=MonitoringAPI(**monitoring_data.get('api', {})),
                load_testing=MonitoringLoadTesting(**monitoring_data.get('load_testing', {})),
                alerts=MonitoringAlerts(**monitoring_data.get('alerts', {}))
            )
        else:
            # Provide default monitoring configuration
            self.monitoring = MonitoringConfig(
                enabled=True,
                detailed_metrics=True,
                experiment_name='pipeline_scalability_monitoring',
                separate_experiment=True,
                resource_monitoring_interval=1.0,
                memory_threshold_warning_mb=8192,
                cpu_threshold_warning_percent=85,
                thresholds=MonitoringThresholds(
                    min_throughput_records_per_second=100,
                    max_processing_time_per_record_ms=100,
                    min_partition_efficiency=0.7,
                    min_scalability_score=0.6,
                    max_memory_usage_percent=85,
                    max_cpu_usage_percent=80
                ),
                generate_reports=True,
                save_metrics_json=True,
                console_output=True,
                dashboard=MonitoringDashboard(
                    enabled=True,
                    port=8501,
                    auto_refresh_seconds=5,
                    historical_data_points=100
                ),
                api=MonitoringAPI(
                    enabled=True,
                    port=8000,
                    cors_origins=["*"]
                ),
                load_testing=MonitoringLoadTesting(
                    data_volume_multipliers=[1, 2, 5, 10],
                    concurrent_users=[1, 5, 10, 20],
                    benchmark_duration_minutes=5
                ),
                alerts=MonitoringAlerts(
                    enabled=True,
                    email_notifications=False,
                    slack_webhook=None,
                    performance_degradation_threshold=0.3
                )
            )
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value


class Logger:
    """Simplified logging utility class"""
    
    def __init__(self):
        self.logger = logging.getLogger("SparkMLPipeline")
    
    @staticmethod
    def setup_logging(config: Config = None):
        """Setup logging configuration"""
        
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(logs_dir / "pipeline.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        return logging.getLogger("SparkMLPipeline")
    
    def get_logger(self):
        """Get logger instance"""
        return self.logger


class MLflowManager:
    """MLflow management utilities"""
    
    def __init__(self, config: Config):
        self.config = config
        self.mlflow_config = config.mlflow_config
        self.logger = Logger().get_logger()
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Setup MLflow configuration"""
        # Set tracking URI from environment or default
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'mlflow_tracking')
        
        # Handle file:// prefix correctly
        if tracking_uri.startswith('file://'):
            mlflow.set_tracking_uri(tracking_uri)
        else:
            mlflow.set_tracking_uri(f"file://{Path(tracking_uri).resolve()}")
        
        # Set experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.mlflow_config['experiment_name'])
            if experiment is None:
                mlflow.create_experiment(self.mlflow_config['experiment_name'])
        except Exception as e:
            self.logger.warning(f"MLflow experiment setup warning: {e}")
        
        mlflow.set_experiment(self.mlflow_config['experiment_name'])
    
    def log_config(self, config: Config):
        """Log configuration parameters to MLflow"""
        if self.mlflow_config['log_params']:
            mlflow.log_params({
                'sequence_length': config.data.sequence_length,
                'train_split': config.data.train_split,
                'algorithm': config.model.algorithm,
                **config.model.default_params
            })
    
    def log_feature_config(self, feature_names: List[str]):
        """Log feature configuration"""
        if self.mlflow_config['log_params']:
            mlflow.log_param('num_features', len(feature_names))
            mlflow.log_param('feature_names', ','.join(feature_names))
    
    def register_model(self, model, model_name: Optional[str] = None):
        """Register model to MLflow Model Registry"""
        if not self.mlflow_config['log_models']:
            return
        
        model_name = model_name or self.mlflow_config['model_name']
        
        try:
            # Log model
            mlflow.spark.log_model(model, "model")
            
            # Register model
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            mlflow.register_model(model_uri, model_name)
            
            self.logger.info(f"Model registered as: {model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to register model: {e}")


class PathManager:
    """Path management utilities"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.ensure_directories()
    
    def ensure_directories(self):
        """Ensure all necessary directories exist"""
        directories = [
            "data_csv",
            "delta_tables",
            "models",
            "results/training",
            "results/inference", 
            "results/evaluation",
            "logs",
            "plots",
            "checkpoints",
            "mlflow_tracking"
        ]
        
        for directory in directories:
            (self.base_dir / directory).mkdir(parents=True, exist_ok=True)
    
    @property
    def data_csv_dir(self) -> Path:
        return self.base_dir / "data_csv"
    
    @property
    def delta_tables_dir(self) -> Path:
        return self.base_dir / "delta_tables"
    
    @property
    def models_dir(self) -> Path:
        return self.base_dir / "models"
    
    @property
    def results_dir(self) -> Path:
        return self.base_dir / "results"
    
    @property
    def plots_dir(self) -> Path:
        return self.base_dir / "plots"
    
    @property
    def logs_dir(self) -> Path:
        return self.base_dir / "logs"
    
    @property
    def mlflow_tracking_dir(self) -> Path:
        return self.base_dir / "mlflow_tracking"
    
    def get_csv_files(self) -> List[Path]:
        """Get list of CSV files"""
        return list(self.data_csv_dir.glob("*.csv"))


def load_environment():
    """Load environment variables from .env file"""
    env_file = Path('.env')
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)


# Global configuration instance
def get_config() -> Config:
    """Get global configuration instance"""
    if not hasattr(get_config, '_instance'):
        get_config._instance = Config()
    return get_config._instance