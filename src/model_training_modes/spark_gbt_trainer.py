#!/usr/bin/env python3
"""
Spark ML GBT Trainer - Enhanced Current Implementation
Exactly like current implementation with enhanced monitoring for scalability evaluation

Part of Data Engineering at Scale project - Mode 1: Spark GBT
"""

import time
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple
from pathlib import Path

import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient

from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Hyperparameter tuning imports
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from src.utils import Config, Logger, MLflowManager
from src.data_processing import DataProcessor
from src.model_training import ModelTrainer
from src.visualization import Visualizer


class SparkGBTTrainer:
    """
    Spark ML GBT Trainer with enhanced monitoring
    Exactly like current implementation but with comprehensive scalability tracking
    """
    
    def __init__(self, mode_config: Dict, base_config: Config, spark: SparkSession, monitor=None):
        """
        Initialize Spark GBT trainer
        
        Args:
            mode_config: Mode-specific configuration from config_modes.yaml
            base_config: Base configuration for data and features
            spark: Spark session
            monitor: Scalability monitor instance (optional)
        """
        self.mode_config = mode_config
        self.base_config = base_config
        self.spark = spark
        self.monitor = monitor
        self.logger = Logger().get_logger()
        
        # MLflow setup (must be before _create_training_config)
        self.experiment_name = mode_config['mlflow']['experiment_name']
        self.model_name = mode_config['mlflow']['model_name']
        self.run_name_prefix = mode_config['mlflow']['run_name_prefix']
        
        # Initialize components using base config but override model params
        self.data_processor = DataProcessor(base_config, spark)
        
        # Create modified config for model trainer with mode-specific params
        self.training_config = self._create_training_config()
        self.model_trainer = ModelTrainer(self.training_config, spark)
        self.visualizer = Visualizer(base_config)
        
        # Monitoring metrics
        self.training_metrics = {
            'mode': 'spark_gbt',
            'start_time': None,
            'end_time': None,
            'training_time': 0,
            'data_loading_time': 0,
            'feature_engineering_time': 0,
            'model_training_time': 0,
            'evaluation_time': 0,
            'stocks_processed': [],
            'total_samples': 0,
            'driver_memory_usage': [],
            'scalability_specific_metrics': {}
        }
        
        self.logger.info("🌳 Spark GBT Trainer initialized (enhanced monitoring)")
    
    def _create_training_config(self) -> Config:
        """Create training configuration with mode-specific model parameters"""
        
        # Start with base config
        config = self.base_config
        
        # Override model parameters with mode-specific ones
        config.model.algorithm = self.mode_config['model']['algorithm']
        config.model.default_params = self.mode_config['model']['default_params']
        config.model.hyperparameter_ranges = self.mode_config['model']['hyperparameter_ranges']
        
        # Override hyperparameter tuning settings
        config.hyperparameter_tuning['enabled'] = self.mode_config['hyperparameter_tuning']['enabled']
        config.hyperparameter_tuning['method'] = self.mode_config['hyperparameter_tuning']['method']
        config.hyperparameter_tuning['n_trials'] = self.mode_config['hyperparameter_tuning']['n_trials']
        config.hyperparameter_tuning['cv_folds'] = self.mode_config['hyperparameter_tuning']['cv_folds']
        config.hyperparameter_tuning['direction'] = self.mode_config['hyperparameter_tuning']['direction']
        config.hyperparameter_tuning['metric'] = self.mode_config['hyperparameter_tuning']['metric']
        
        # Override MLflow settings
        config.mlflow_config['experiment_name'] = self.experiment_name
        config.mlflow_config['model_name'] = self.model_name
        config.mlflow_config['run_name_prefix'] = self.run_name_prefix
        
        return config
    
    def train_unified_model(self, stock_selection: List[str], 
                          enable_scalability_eval: bool = False) -> Dict:
        """
        Train unified GBT model with enhanced monitoring
        Exactly like current implementation with added scalability tracking
        
        Args:
            stock_selection: List of stock symbols to train on
            enable_scalability_eval: Whether to enable detailed scalability evaluation
            
        Returns:
            Dict containing model info, metrics, and scalability data
        """
        
        start_time = time.time()
        self.training_metrics['start_time'] = datetime.now()
        self.training_metrics['stocks_processed'] = stock_selection
        
        self.logger.info(f"🌳 Starting Spark GBT unified training with {len(stock_selection)} stocks")
        self.logger.info(f"📊 Scalability evaluation: {'ENABLED' if enable_scalability_eval else 'DISABLED'}")
        
        try:
            # Set up MLflow experiment
            self._setup_mlflow_experiment()
            
            # Phase 1: Data Loading and Preprocessing (with timing)
            data_load_start = time.time()
            self.logger.info("📊 Phase 1: Loading and preprocessing data...")
            
            train_df, test_df, feature_names, scaler_model = self.data_processor.prepare_training_data(
                stock_selection=stock_selection
            )
            
            training_samples = train_df.count()
            test_samples = test_df.count()
            total_samples = training_samples + test_samples
            
            self.training_metrics['data_loading_time'] = time.time() - data_load_start
            self.training_metrics['total_samples'] = total_samples
            
            self.logger.info(f"✅ Data loaded: {total_samples:,} total samples ({training_samples:,} train, {test_samples:,} test)")
            self.logger.info(f"⏱️ Data loading time: {self.training_metrics['data_loading_time']:.2f}s")
            
            # Monitor data volume for scalability metrics
            if self.monitor:
                self.monitor.monitor_data_volume("spark_gbt_data_processing")
            
            # Phase 2: Feature Engineering (timing already tracked in data processor)
            feature_eng_start = time.time()
            self.training_metrics['feature_engineering_time'] = time.time() - feature_eng_start
            
            self.logger.info(f"🔧 Features created: {len(feature_names)}")
            
            # Phase 3: Model Training (with enhanced monitoring)
            model_training_start = time.time()
            self.logger.info("🌳 Phase 2: Training GBT model...")
            
            # Create unified model name
            if len(stock_selection) <= 3:
                model_name = f"{self.model_name}_{'_'.join(sorted(stock_selection))}"
            else:
                model_name = f"{self.model_name}_unified_{len(stock_selection)}stocks"
            
            # Train model with MLflow tracking
            model_result = self._train_gbt_model_with_monitoring(
                train_df, test_df, feature_names, scaler_model, model_name, stock_selection
            )
            
            self.training_metrics['model_training_time'] = time.time() - model_training_start
            
            # Phase 4: Evaluation and Visualization
            eval_start = time.time()
            self.logger.info("📊 Phase 3: Creating visualizations...")
            
            # Visualizations are created inside MLflow run context in _train_gbt_model_with_monitoring
            
            self.training_metrics['evaluation_time'] = time.time() - eval_start
            
            # Calculate total training time
            total_training_time = time.time() - start_time
            self.training_metrics['training_time'] = total_training_time
            self.training_metrics['end_time'] = datetime.now()
            
            # Record performance data to monitoring system
            if self.monitor:
                self.monitor.monitor_performance(
                    operation_name='spark_gbt_training',
                    start_time=start_time,
                    records_processed=training_samples + test_samples
                )
            
            # Collect scalability-specific metrics if enabled
            if enable_scalability_eval:
                self.training_metrics['scalability_specific_metrics'] = self._collect_scalability_metrics(
                    total_samples, total_training_time
                )
            
            # Prepare result
            result = {
                'mode': 'spark_gbt',
                'model_info': model_result['model_info'],
                'metrics': model_result['metrics'],
                'training_time': total_training_time,
                'training_samples': training_samples,
                'test_samples': test_samples,
                'features_count': len(feature_names),
                'stocks_processed': stock_selection,
                'phase_timings': {
                    'data_loading': self.training_metrics['data_loading_time'],
                    'feature_engineering': self.training_metrics['feature_engineering_time'],
                    'model_training': self.training_metrics['model_training_time'],
                    'evaluation': self.training_metrics['evaluation_time']
                },
                'scalability_metrics': self.training_metrics['scalability_specific_metrics'] if enable_scalability_eval else None
            }
            
            self.logger.info(f"✅ Spark GBT training completed in {total_training_time:.2f} seconds")
            self.logger.info(f"📈 Model performance (R²): {model_result['metrics']['test_metrics']['r2']:.4f}")
            
            return result
            
        except Exception as e:
            error_msg = f"Spark GBT training failed: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _setup_mlflow_experiment(self):
        """Set up MLflow experiment for Spark GBT mode"""
        
        try:
            experiment_id = mlflow.create_experiment(self.experiment_name)
            self.logger.info(f"📂 Created new MLflow experiment: {self.experiment_name}")
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            experiment_id = experiment.experiment_id
            self.logger.info(f"📂 Using existing MLflow experiment: {self.experiment_name}")
        
        mlflow.set_experiment(self.experiment_name)
    
    def _train_gbt_model_with_monitoring(self, train_df: DataFrame, test_df: DataFrame, 
                                       feature_names: List[str], scaler_model, model_name: str,
                                       stock_selection: List[str]) -> Dict:
        """
        Train GBT model with enhanced monitoring (exactly like current implementation)
        """
        
        # End any active runs to avoid conflicts
        if mlflow.active_run():
            mlflow.end_run()
        
        run_name = f"{self.run_name_prefix}_unified_monitoring_{'-'.join(sorted(stock_selection))}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name) as run:
            
            # Log unified model parameters
            mlflow.log_param("mode", "spark_gbt")
            mlflow.log_param("algorithm", "GBTRegressor") 
            mlflow.log_param("stocks_included", ','.join(sorted(stock_selection)))
            mlflow.log_param("num_stocks", len(stock_selection))
            mlflow.log_param("training_samples", train_df.count())
            mlflow.log_param("test_samples", test_df.count())
            mlflow.log_param("features_count", len(feature_names))
            
            # Log individual stock sample counts
            for stock in stock_selection:
                stock_train_count = train_df.filter(train_df.stock_symbol == stock).count()
                stock_test_count = test_df.filter(test_df.stock_symbol == stock).count()
                mlflow.log_param(f"{stock}_training_samples", stock_train_count)
                mlflow.log_param(f"{stock}_test_samples", stock_test_count)
            
            # Log configuration using MLflow manager
            mlflow_manager = MLflowManager(self.training_config)
            mlflow_manager.log_config(self.training_config)
            mlflow_manager.log_feature_config(feature_names)
            
            # Get best hyperparameters
            if self.training_config.hyperparameter_tuning['enabled']:
                self.logger.info("🔍 Hyperparameter tuning is ENABLED")
                best_params = self.model_trainer._tune_hyperparameters(train_df, test_df)
            else:
                self.logger.info("⚙️ Using default parameters")
                best_params = self.training_config.model.default_params
            
            # Train final model
            model = self.model_trainer._train_final_model(train_df, best_params)
            
            # Evaluate model
            evaluation_results = self.model_trainer._evaluate_model(model, train_df, test_df, feature_names)
            
            # Log metrics
            self.model_trainer._log_metrics(evaluation_results)
            
            # Log model to MLflow
            mlflow.spark.log_model(model, "model")
            
            # Log feature scaler to MLflow
            import tempfile
            scaler_temp_path = tempfile.mktemp(suffix="_feature_scaler")
            scaler_model.write().overwrite().save(scaler_temp_path)
            mlflow.log_artifact(scaler_temp_path, "preprocessing")
            self.logger.info("✅ Feature scaler logged to MLflow")
            
            # Register model with versioning
            model_version = mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/model",
                name=model_name
            ).version
            
            # Tag model for tracking
            client = MlflowClient()
            client.set_model_version_tag(
                name=model_name,
                version=model_version,
                key="mode",
                value="spark_gbt"
            )
            client.set_model_version_tag(
                name=model_name,
                version=model_version,
                key="stocks_included",
                value=','.join(sorted(stock_selection))
            )
            client.set_model_version_tag(
                name=model_name,
                version=model_version,
                key="training_date",
                value=datetime.now().isoformat()
            )
            
            model_path = f"models:/{model_name}/{model_version}"
            model_info = {
                'name': model_name,
                'version': model_version,
                'path': model_path,
                'type': 'spark_gbt'
            }
            
            # Create visualizations (same types for comparison across modes)
            try:
                self.model_trainer._create_visualizations(
                    model, train_df, test_df, feature_names, evaluation_results
                )
                self.logger.info("✅ Visualizations created and logged to MLflow")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to create visualizations: {e}")
            
            self.logger.info(f"💾 Spark GBT model saved: {model_path}")
            
            return {
                'model_info': model_info,
                'metrics': evaluation_results,
                'mlflow_run_id': run.info.run_id,
                'mlflow_experiment_name': self.experiment_name
            }
    
    def _collect_scalability_metrics(self, total_samples: int, total_training_time: float) -> Dict:
        """Collect scalability-specific metrics for GBT mode"""
        
        import psutil
        
        # Driver-focused metrics (since GBT trains on driver)
        driver_memory = psutil.virtual_memory()
        driver_cpu_percent = psutil.cpu_percent()
        
        scalability_metrics = {
            'mode_specific_focus': 'driver_based_training',
            'throughput_samples_per_second': total_samples / total_training_time if total_training_time > 0 else 0,
            'driver_memory_usage_mb': driver_memory.used / (1024 * 1024),
            'driver_memory_percent': driver_memory.percent,
            'driver_cpu_percent': driver_cpu_percent,
            'training_efficiency_score': self._calculate_gbt_efficiency_score(total_samples, total_training_time),
            'sequential_training_characteristics': {
                'benefits': 'Efficient for moderate datasets, interpretable feature importance',
                'limitations': 'Sequential training, limited parallelism in GBT algorithm',
                'optimal_use_case': 'High accuracy requirements with moderate data volumes'
            }
        }
        
        return scalability_metrics
    
    def _calculate_gbt_efficiency_score(self, total_samples: int, total_training_time: float) -> float:
        """Calculate GBT-specific efficiency score"""
        
        if total_samples == 0 or total_training_time == 0:
            return 0.0
        
        # Expected baseline for GBT: ~1000 samples per second for moderate complexity
        baseline_throughput = 1000.0
        actual_throughput = total_samples / total_training_time
        
        efficiency_score = min(1.0, actual_throughput / baseline_throughput)
        
        return efficiency_score
    
    def get_training_summary(self) -> Dict:
        """Get training summary for reporting"""
        
        return {
            'mode': 'spark_gbt',
            'description': 'Spark ML GBT with driver-based training (enhanced current implementation)',
            'characteristics': {
                'training_type': 'sequential_on_driver',
                'parallelism': 'limited_gbt_internal',
                'memory_focus': 'driver_memory_intensive',
                'scalability_pattern': 'linear_with_data_volume'
            },
            'monitoring_focus': [
                'driver_memory_usage',
                'sequential_training_time',
                'feature_importance_analysis', 
                'tree_depth_optimization'
            ],
            'metrics': self.training_metrics
        }