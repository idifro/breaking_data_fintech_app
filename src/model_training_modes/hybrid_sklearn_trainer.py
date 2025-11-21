#!/usr/bin/env python3
"""
Hybrid Sklearn Trainer - Spark Preprocessing + Sklearn Training + Distributed Inference
Collect features to driver for sklearn training, then distributed inference using pandas UDF

Part of Data Engineering at Scale project - Mode 3: Hybrid Sklearn
"""

import time
import os
import pickle
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, pandas_udf, struct
from pyspark.sql.types import DoubleType
import pyspark.sql.functions as F

from src.utils import Config, Logger
from src.data_processing import DataProcessor
from src.visualization import Visualizer


class HybridSklearnTrainer:
    """
    Hybrid Sklearn Trainer with Spark preprocessing and distributed inference
    Collects features to driver for pandas/sklearn training, then distributed inference
    """
    
    def __init__(self, mode_config: Dict, base_config: Config, spark: SparkSession, monitor=None):
        """
        Initialize Hybrid Sklearn trainer
        
        Args:
            mode_config: Mode-specific configuration from config_modes.yaml
            base_config: Base configuration for data and features
            spark: Spark session
            monitor: ScalabilityMonitor instance for recording performance data
        """
        self.mode_config = mode_config
        self.base_config = base_config
        self.spark = spark
        self.monitor = monitor
        self.logger = Logger().get_logger()
        
        # Initialize components
        self.data_processor = DataProcessor(base_config, spark)
        self.visualizer = Visualizer(base_config)
        
        # MLflow setup
        self.experiment_name = mode_config['mlflow']['experiment_name']
        self.model_name = mode_config['mlflow']['model_name']
        self.run_name_prefix = mode_config['mlflow']['run_name_prefix']
        
        # Sklearn model and scaler
        self.sklearn_model = None
        self.feature_scaler = None
        
        # Memory optimization tracking
        self.data_sampling_ratio = None
        
        # Monitoring metrics
        self.training_metrics = {
            'mode': 'hybrid_sklearn',
            'start_time': None,
            'end_time': None,
            'training_time': 0,
            'data_loading_time': 0,
            'feature_engineering_time': 0,
            'data_collection_time': 0,
            'sklearn_training_time': 0,
            'distributed_inference_time': 0,
            'evaluation_time': 0,
            'stocks_processed': [],
            'total_samples': 0,
            'driver_sklearn_metrics': {},
            'distributed_inference_metrics': {},
            'scalability_specific_metrics': {}
        }
        
        self.logger.info("🔄 Hybrid Sklearn Trainer initialized (Spark preprocessing + sklearn training + distributed inference)")
    
    def train_unified_model(self, stock_selection: List[str], 
                          enable_scalability_eval: bool = False) -> Dict:
        """
        Train unified model using hybrid approach
        
        Args:
            stock_selection: List of stock symbols to train on
            enable_scalability_eval: Whether to enable detailed scalability evaluation
            
        Returns:
            Dict containing model info, metrics, and scalability data
        """
        
        start_time = time.time()
        self.training_metrics['start_time'] = datetime.now()
        self.training_metrics['stocks_processed'] = stock_selection
        
        self.logger.info(f"🔄 Starting Hybrid Sklearn training with {len(stock_selection)} stocks")
        self.logger.info(f"📊 Scalability evaluation: {'ENABLED' if enable_scalability_eval else 'DISABLED'}")
        
        try:
            # Set up MLflow experiment
            self._setup_mlflow_experiment()
            
            # Phase 1: Spark Data Loading and Preprocessing
            data_load_start = time.time()
            self.logger.info("📊 Phase 1: Spark data loading and preprocessing...")
            
            train_df, test_df, feature_names, spark_scaler_model = self.data_processor.prepare_training_data(
                stock_selection=stock_selection
            )
            
            training_samples = train_df.count()
            test_samples = test_df.count()
            total_samples = training_samples + test_samples
            
            self.training_metrics['data_loading_time'] = time.time() - data_load_start
            self.training_metrics['total_samples'] = total_samples
            
            self.logger.info(f"✅ Spark preprocessing completed: {total_samples:,} total samples")
            self.logger.info(f"⏱️ Data loading time: {self.training_metrics['data_loading_time']:.2f}s")
            
            # Phase 2: Data Collection to Driver
            collection_start = time.time()
            self.logger.info("🔄 Phase 2: Collecting features to driver for sklearn training...")
            
            # Check data size before collection
            self._validate_collection_size(total_samples, len(feature_names))
            
            # Collect data to driver as pandas DataFrames
            train_pandas_df, test_pandas_df = self._collect_data_to_driver(
                train_df, test_df, feature_names
            )
            
            self.training_metrics['data_collection_time'] = time.time() - collection_start
            
            self.logger.info(f"✅ Data collected to driver: {len(train_pandas_df)} train, {len(test_pandas_df)} test samples")
            self.logger.info(f"⏱️ Collection time: {self.training_metrics['data_collection_time']:.2f}s")
            
            # Monitor data volume for scalability metrics
            if self.monitor:
                self.monitor.monitor_data_volume("hybrid_sklearn_data_processing")
            
            # Phase 3: Sklearn Training on Driver
            sklearn_training_start = time.time()
            self.logger.info("🤖 Phase 3: Sklearn model training on driver...")
            
            # Create unified model name
            if len(stock_selection) <= 3:
                model_name = f"{self.model_name}_{'_'.join(sorted(stock_selection))}"
            else:
                model_name = f"{self.model_name}_unified_{len(stock_selection)}stocks"
            
            # Train sklearn model
            model_result = self._train_sklearn_model_with_monitoring(
                train_pandas_df, test_pandas_df, feature_names, spark_scaler_model, model_name, stock_selection
            )
            
            self.training_metrics['sklearn_training_time'] = time.time() - sklearn_training_start
            
            # Phase 4: Distributed Inference
            inference_start = time.time()
            self.logger.info("🌐 Phase 4: Distributed inference using pandas UDF...")
            
            # Create distributed predictions
            train_predictions_df, test_predictions_df = self._create_distributed_predictions(
                train_df, test_df, feature_names
            )
            
            self.training_metrics['distributed_inference_time'] = time.time() - inference_start
            
            # Phase 5: Evaluation and Visualization
            eval_start = time.time()
            self.logger.info("📊 Phase 5: Creating visualizations...")
            
            # Create visualizations using distributed predictions
            evaluation_results = self._evaluate_hybrid_model(
                train_predictions_df, test_predictions_df, feature_names
            )
            
            self.training_metrics['evaluation_time'] = time.time() - eval_start
            
            # Calculate total training time
            total_training_time = time.time() - start_time
            self.training_metrics['training_time'] = total_training_time
            self.training_metrics['end_time'] = datetime.now()
            
            # Record performance data to monitoring system
            if self.monitor:
                self.monitor.monitor_performance(
                    operation_name='hybrid_sklearn_training',
                    start_time=start_time,
                    records_processed=training_samples + test_samples
                )
            
            # Always collect scalability-specific metrics for proper monitoring
            self.training_metrics['scalability_specific_metrics'] = self._collect_hybrid_scalability_metrics(
                total_samples, total_training_time
            )
            
            # Collect additional detailed metrics if enabled
            if enable_scalability_eval:
                self.training_metrics['scalability_specific_metrics'].update({
                    'detailed_evaluation_enabled': True,
                    'additional_metrics': 'enhanced_monitoring_mode'
                })
            
            # Update model result with evaluation
            model_result['metrics'] = evaluation_results
            
            # Prepare result
            result = {
                'mode': 'hybrid_sklearn',
                'model_info': model_result['model_info'],
                'metrics': evaluation_results,
                'training_time': total_training_time,
                'training_samples': training_samples,
                'test_samples': test_samples,
                'features_count': len(feature_names),
                'stocks_processed': stock_selection,
                'phase_timings': {
                    'data_loading': self.training_metrics['data_loading_time'],
                    'data_collection': self.training_metrics['data_collection_time'],
                    'sklearn_training': self.training_metrics['sklearn_training_time'],
                    'distributed_inference': self.training_metrics['distributed_inference_time'],
                    'evaluation': self.training_metrics['evaluation_time']
                },
                'hybrid_metrics': self.training_metrics['distributed_inference_metrics'],
                'scalability_metrics': self.training_metrics['scalability_specific_metrics']
            }
            
            self.logger.info(f"✅ Hybrid sklearn training completed in {total_training_time:.2f} seconds")
            self.logger.info(f"📈 Model performance (R²): {evaluation_results['test_metrics']['r2']:.4f}")
            
            return result
            
        except Exception as e:
            error_msg = f"Hybrid sklearn training failed: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _setup_mlflow_experiment(self):
        """Set up MLflow experiment for Hybrid Sklearn mode"""
        
        try:
            experiment_id = mlflow.create_experiment(self.experiment_name)
            self.logger.info(f"📂 Created new MLflow experiment: {self.experiment_name}")
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            experiment_id = experiment.experiment_id
            self.logger.info(f"📂 Using existing MLflow experiment: {self.experiment_name}")
        
        mlflow.set_experiment(self.experiment_name)
    
    def _validate_collection_size(self, total_samples: int, num_features: int):
        """Validate that data size is appropriate for collection to driver and implement optimizations"""
        
        # Estimate memory requirement
        bytes_per_sample = num_features * 8 + 50  # 8 bytes per float + overhead
        estimated_memory_mb = (total_samples * bytes_per_sample) / (1024 * 1024)
        
        max_collect_size_mb = self.mode_config['hybrid_config']['max_collect_size_mb']
        
        self.logger.info(f"💾 Estimated collection size: {estimated_memory_mb:.1f} MB")
        
        # Implement adaptive memory management
        if estimated_memory_mb > max_collect_size_mb:
            self.logger.warning(f"⚠️ Data size ({estimated_memory_mb:.1f} MB) exceeds max collection size ({max_collect_size_mb} MB)")
            
            # Calculate sampling ratio to stay within memory limits
            sampling_ratio = min(0.8, max_collect_size_mb / estimated_memory_mb)
            self.data_sampling_ratio = sampling_ratio
            
            self.logger.warning(f"� MEMORY OPTIMIZATION: Will sample {sampling_ratio:.1%} of data to prevent system slowdown")
            self.logger.warning(f"📊 Reduced data size: ~{estimated_memory_mb * sampling_ratio:.1f} MB")
            
            # Log optimization strategy
            self.training_metrics['memory_optimization'] = {
                'original_size_mb': estimated_memory_mb,
                'max_allowed_mb': max_collect_size_mb,
                'sampling_ratio': sampling_ratio,
                'optimized_size_mb': estimated_memory_mb * sampling_ratio,
                'reason': 'prevent_system_memory_pressure'
            }
        else:
            self.data_sampling_ratio = None
            self.logger.info(f"✅ Memory usage within limits ({estimated_memory_mb:.1f} MB < {max_collect_size_mb} MB)")
    
    def _collect_data_to_driver(self, train_df: DataFrame, test_df: DataFrame, 
                               feature_names: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Collect Spark DataFrames to driver as pandas DataFrames with memory optimization"""
        
        # Select required columns (include stock_symbol if available)
        feature_cols = feature_names + ["target"]
        if "stock_symbol" in train_df.columns:
            feature_cols.append("stock_symbol")
        
        # Apply sampling if memory optimization is needed
        if hasattr(self, 'data_sampling_ratio') and self.data_sampling_ratio is not None:
            self.logger.info(f"🔧 Applying memory optimization sampling: {self.data_sampling_ratio:.1%}")
            
            # Sample both training and test data proportionally
            train_df_sampled = train_df.sample(fraction=self.data_sampling_ratio, seed=42)
            test_df_sampled = test_df.sample(fraction=self.data_sampling_ratio, seed=42)
            
            self.logger.info(f"📊 Sampled training data: {train_df_sampled.count():,} samples (was {train_df.count():,})")
            self.logger.info(f"📊 Sampled test data: {test_df_sampled.count():,} samples (was {test_df.count():,})")
            
            # Use sampled data for collection
            train_df = train_df_sampled
            test_df = test_df_sampled
        
        self.logger.info("🔄 Collecting training data to driver...")
        
        # Optimize collection with partitioning and caching
        train_df.cache()
        test_df.cache()
        
        # Collect training data with memory management
        train_pandas_df = train_df.select(*feature_cols).toPandas()
        
        self.logger.info("🔄 Collecting test data to driver...")
        
        # Collect test data
        test_pandas_df = test_df.select(*feature_cols).toPandas()
        
        # Unpersist cached DataFrames to free Spark memory
        train_df.unpersist()
        test_df.unpersist()
        
        self.logger.info(f"✅ Data collection completed - Train: {len(train_pandas_df)}, Test: {len(test_pandas_df)}")
        
        # Force garbage collection to free memory
        import gc
        gc.collect()
        
        return train_pandas_df, test_pandas_df
    
    def _train_sklearn_model_with_monitoring(self, train_pandas_df: pd.DataFrame, 
                                           test_pandas_df: pd.DataFrame,
                                           feature_names: List[str], spark_scaler_model,
                                           model_name: str, stock_selection: List[str]) -> Dict:
        """
        Train sklearn model on driver with MLflow tracking
        """
        
        # End any active runs to avoid conflicts
        if mlflow.active_run():
            mlflow.end_run()
        
        run_name = f"{self.run_name_prefix}_hybrid_monitoring_{'-'.join(sorted(stock_selection))}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name) as run:
            
            # Ensure all needed imports are available for the entire method
            import numpy as np
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
            
            # Log unified model parameters
            mlflow.log_param("mode", "hybrid_sklearn")
            mlflow.log_param("algorithm", self.mode_config['model']['algorithm'])
            mlflow.log_param("training_approach", "sklearn_on_driver_distributed_inference")
            mlflow.log_param("stocks_included", ','.join(sorted(stock_selection)))
            mlflow.log_param("num_stocks", len(stock_selection))
            mlflow.log_param("training_samples", len(train_pandas_df))
            mlflow.log_param("test_samples", len(test_pandas_df))
            mlflow.log_param("features_count", len(feature_names))
            
            # Log hybrid approach details
            mlflow.log_param("data_collection_to_driver", True)
            mlflow.log_param("distributed_inference", True)
            mlflow.log_param("sklearn_framework", "pandas_sklearn")
            
            # Prepare features and target
            X_train = train_pandas_df[feature_names]
            y_train = train_pandas_df['target']
            X_test = test_pandas_df[feature_names]
            y_test = test_pandas_df['target']
            
            # Feature scaling
            self.logger.info("🔧 Applying feature scaling...")
            self.feature_scaler = StandardScaler()
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_test_scaled = self.feature_scaler.transform(X_test)
            
            # Get best hyperparameters for sklearn model
            if self.mode_config['hyperparameter_tuning']['enabled']:
                self.logger.info("🔍 Sklearn hyperparameter tuning is ENABLED")
                best_params = self._tune_sklearn_hyperparameters(X_train_scaled, y_train)
            else:
                self.logger.info("⚙️ Using default sklearn parameters")
                best_params = self.mode_config['model']['default_params']
            
            # Log best parameters
            for param, value in best_params.items():
                mlflow.log_param(f"best_{param}", value)
            
            # Create and train sklearn model
            self.logger.info(f"🤖 Training {self.mode_config['model']['algorithm']} on driver...")
            
            if self.mode_config['model']['algorithm'] == 'RandomForestRegressor':
                self.sklearn_model = RandomForestRegressor(**best_params)
            elif self.mode_config['model']['algorithm'] == 'GradientBoostingRegressor':
                self.sklearn_model = GradientBoostingRegressor(**best_params)
            else:
                raise ValueError(f"Unsupported sklearn algorithm: {self.mode_config['model']['algorithm']}")
            
            # Train model
            sklearn_fit_start = time.time()
            self.sklearn_model.fit(X_train_scaled, y_train)
            sklearn_fit_time = time.time() - sklearn_fit_start
            
            mlflow.log_metric("sklearn_fit_time_seconds", sklearn_fit_time)
            
            # Make predictions for evaluation
            y_train_pred = self.sklearn_model.predict(X_train_scaled)
            y_test_pred = self.sklearn_model.predict(X_test_scaled)
            
            # Calculate sklearn-based metrics
            train_metrics = {
                'mae': mean_absolute_error(y_train, y_train_pred),
                'mse': mean_squared_error(y_train, y_train_pred),
                'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'r2': r2_score(y_train, y_train_pred)
            }
            
            test_metrics = {
                'mae': mean_absolute_error(y_test, y_test_pred),
                'mse': mean_squared_error(y_test, y_test_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'r2': r2_score(y_test, y_test_pred)
            }
            
            # Log sklearn metrics
            for metric_name, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", value)
            
            for metric_name, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", value)
            
            # Log sklearn model
            mlflow.sklearn.log_model(self.sklearn_model, "sklearn_model")
            
            # Log scalers and additional metrics
            # 1. Sklearn scaler (for local sklearn training)
            with open("/tmp/feature_scaler.pkl", "wb") as f:
                pickle.dump(self.feature_scaler, f)
            mlflow.log_artifact("/tmp/feature_scaler.pkl", "preprocessing")
            
            # 2. Spark scaler (for distributed inference)
            import tempfile
            spark_scaler_temp_path = tempfile.mktemp(suffix="_spark_scaler")
            spark_scaler_model.write().overwrite().save(spark_scaler_temp_path)
            mlflow.log_artifact(spark_scaler_temp_path, "preprocessing")
            self.logger.info("✅ Both sklearn and Spark scalers logged to MLflow")
            
            # Log training phase metrics
            mlflow.log_metric("sklearn_fit_time_seconds", sklearn_fit_time)
            mlflow.log_metric("data_collection_time", self.training_metrics['data_collection_time'])
            mlflow.log_metric("distributed_inference_time", self.training_metrics['distributed_inference_time'])
            
            # Log phase timings as metrics
            for phase, timing in {
                'data_loading_time': self.training_metrics['data_loading_time'],
                'data_collection_time': self.training_metrics['data_collection_time'],
                'sklearn_training_time': sklearn_fit_time,
                'distributed_inference_time': self.training_metrics['distributed_inference_time']
            }.items():
                mlflow.log_metric(phase, timing)
            
            # Create model registry entry (placeholder for sklearn model)
            # Note: We'll create a Spark-compatible wrapper for inference
            model_version = mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/sklearn_model",
                name=model_name
            ).version
            
            # Create and log visualizations within MLflow run
            self.logger.info("📊 Creating visualizations within MLflow run...")
            try:
                import pandas as pd
                
                # Create feature importance plot
                feature_names_list = list(feature_names)
                if hasattr(self.sklearn_model, 'feature_importances_'):
                    importance_values = self.sklearn_model.feature_importances_
                elif hasattr(self.sklearn_model, 'coef_'):
                    importance_values = np.abs(self.sklearn_model.coef_)
                else:
                    importance_values = np.ones(len(feature_names_list)) / len(feature_names_list)
                
                # Create feature importance dataframe
                importance_df = pd.DataFrame({
                    'feature': feature_names_list,
                    'importance': importance_values
                }).sort_values('importance', ascending=False)
                
                # Create and log feature importance plot
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(12, 8))
                top_features = importance_df.head(20)
                ax.barh(range(len(top_features)), top_features['importance'])
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels(top_features['feature'])
                ax.set_xlabel('Feature Importance')
                ax.set_title('Top 20 Feature Importance - Hybrid Sklearn Model')
                plt.tight_layout()
                mlflow.log_figure(fig, "feature_importance.png")
                plt.close()
                
                # Create predictions vs actual plot
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.scatter(y_test, y_test_pred, alpha=0.6)
                min_val = min(y_test.min(), y_test_pred.min())
                max_val = max(y_test.max(), y_test_pred.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
                ax.set_xlabel('Actual Returns')
                ax.set_ylabel('Predicted Returns')
                ax.set_title('Predictions vs Actual - Hybrid Sklearn')
                r2 = r2_score(y_test, y_test_pred)
                ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                plt.tight_layout()
                mlflow.log_figure(fig, "predictions_vs_actual.png")
                plt.close()
                
                # Create residuals plot
                residuals = y_test - y_test_pred
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.scatter(y_test_pred, residuals, alpha=0.6)
                ax.axhline(y=0, color='r', linestyle='--', alpha=0.8)
                ax.set_xlabel('Predicted Values')
                ax.set_ylabel('Residuals')
                ax.set_title('Residuals vs Predicted - Hybrid Sklearn')
                plt.tight_layout()
                mlflow.log_figure(fig, "residuals_plot.png")
                plt.close()
                
                # Create per-stock performance plot using sklearn model predictions
                # Calculate per-stock metrics from sklearn predictions
                stock_metrics = {}
                test_df_with_stock = test_pandas_df.copy()
                test_df_with_stock['prediction'] = y_test_pred
                test_df_with_stock['residual'] = test_df_with_stock['target'] - test_df_with_stock['prediction']
                
                # Group by stock if stock_symbol column exists, otherwise use dummy metric
                if 'stock_symbol' in test_df_with_stock.columns:
                    for stock in test_df_with_stock['stock_symbol'].unique():
                        stock_data = test_df_with_stock[test_df_with_stock['stock_symbol'] == stock]
                        stock_metrics[stock] = {
                            'mae': mean_absolute_error(stock_data['target'], stock_data['prediction']),
                            'mse': mean_squared_error(stock_data['target'], stock_data['prediction']),
                            'r2': r2_score(stock_data['target'], stock_data['prediction']),
                            'sample_count': len(stock_data)
                        }
                else:
                    # If no stock_symbol, create overall metrics
                    stock_metrics['Overall'] = {
                        'mae': mean_absolute_error(y_test, y_test_pred),
                        'mse': mean_squared_error(y_test, y_test_pred),
                        'r2': r2_score(y_test, y_test_pred),
                        'sample_count': len(y_test)
                    }
                
                # Create per-stock performance plot
                if stock_metrics:
                    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                    
                    stocks = list(stock_metrics.keys())
                    mae_values = [stock_metrics[stock]['mae'] for stock in stocks]
                    mse_values = [stock_metrics[stock]['mse'] for stock in stocks]
                    r2_values = [stock_metrics[stock]['r2'] for stock in stocks]
                    sample_counts = [stock_metrics[stock]['sample_count'] for stock in stocks]
                    
                    # MAE by stock
                    axes[0, 0].bar(stocks, mae_values)
                    axes[0, 0].set_ylabel('Mean Absolute Error')
                    axes[0, 0].set_title('MAE by Stock - Hybrid Sklearn')
                    axes[0, 0].tick_params(axis='x', rotation=45)
                    
                    # MSE by stock  
                    axes[0, 1].bar(stocks, mse_values)
                    axes[0, 1].set_ylabel('Mean Squared Error')
                    axes[0, 1].set_title('MSE by Stock - Hybrid Sklearn')
                    axes[0, 1].tick_params(axis='x', rotation=45)
                    
                    # R² by stock
                    axes[1, 0].bar(stocks, r2_values)
                    axes[1, 0].set_ylabel('R² Score')
                    axes[1, 0].set_title('R² Score by Stock - Hybrid Sklearn')
                    axes[1, 0].tick_params(axis='x', rotation=45)
                    
                    # Sample count by stock
                    axes[1, 1].bar(stocks, sample_counts)
                    axes[1, 1].set_ylabel('Sample Count')
                    axes[1, 1].set_title('Test Samples by Stock - Hybrid Sklearn')
                    axes[1, 1].tick_params(axis='x', rotation=45)
                    
                    plt.tight_layout()
                    mlflow.log_figure(fig, "per_stock_performance.png")
                    plt.close()
                
                self.logger.info("✅ Visualizations logged to MLflow")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to create MLflow visualizations: {e}")
            
            # Tag model for tracking
            client = MlflowClient()
            client.set_model_version_tag(
                name=model_name,
                version=model_version,
                key="mode",
                value="hybrid_sklearn"
            )
            client.set_model_version_tag(
                name=model_name,
                version=model_version,
                key="training_approach",
                value="sklearn_on_driver"
            )
            client.set_model_version_tag(
                name=model_name,
                version=model_version,
                key="inference_approach",
                value="distributed_pandas_udf"
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
                'type': 'hybrid_sklearn'
            }
            
            # Store driver training metrics
            self.training_metrics['driver_sklearn_metrics'] = {
                'algorithm': self.mode_config['model']['algorithm'],
                'training_time': sklearn_fit_time,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'feature_scaling': True
            }
            
            self.logger.info(f"💾 Hybrid sklearn model saved: {model_path}")
            
            return {
                'model_info': model_info,
                'mlflow_run_id': run.info.run_id,
                'mlflow_experiment_name': self.experiment_name
            }
    
    def _tune_sklearn_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Hyperparameter tuning for sklearn model using GridSearchCV"""
        
        self.logger.info("🔍 Tuning sklearn hyperparameters with GridSearchCV...")
        
        hyperparameter_ranges = self.mode_config['model']['hyperparameter_ranges']
        
        # Create base model
        if self.mode_config['model']['algorithm'] == 'RandomForestRegressor':
            base_model = RandomForestRegressor(random_state=42)
        else:
            base_model = GradientBoostingRegressor(random_state=42)
        
        # Create parameter grid
        param_grid = {}
        for param, values in hyperparameter_ranges.items():
            if param != 'random_state':  # Skip non-tunable params
                param_grid[param] = values[:3]  # Limit for speed
        
        # Perform hyperparameter tuning
        if self.mode_config['hyperparameter_tuning']['method'] == 'grid_search':
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=self.mode_config['hyperparameter_tuning']['cv_folds'],
                scoring=self.mode_config['hyperparameter_tuning']['scoring'],
                n_jobs=-1
            )
        else:
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                cv=self.mode_config['hyperparameter_tuning']['cv_folds'],
                scoring=self.mode_config['hyperparameter_tuning']['scoring'],
                n_iter=self.mode_config['hyperparameter_tuning']['n_iter'],
                n_jobs=-1,
                random_state=42
            )
        
        search.fit(X_train, y_train)
        
        best_params = search.best_params_
        best_params['random_state'] = 42  # Ensure reproducibility
        
        self.logger.info(f"✅ Best sklearn parameters found: {best_params}")
        self.logger.info(f"📊 Best CV score: {search.best_score_:.4f}")
        
        return best_params
    
    def _create_distributed_predictions(self, train_df: DataFrame, test_df: DataFrame,
                                      feature_names: List[str]) -> Tuple[DataFrame, DataFrame]:
        """Create distributed predictions using pandas UDF with memory optimization"""
        
        self.logger.info("🌐 Creating distributed predictions using pandas UDF...")
        
        # Import required vector conversion function
        from pyspark.sql.functions import udf
        from pyspark.sql.types import ArrayType, DoubleType
        from pyspark.ml.linalg import DenseVector, SparseVector
        
        # First, convert feature vectors to arrays using regular UDF
        def vector_to_array(vector):
            """Convert Spark ML Vector to array"""
            if isinstance(vector, DenseVector):
                return vector.toArray().tolist()
            elif isinstance(vector, SparseVector):
                return vector.toArray().tolist()
            else:
                return vector
        
        vector_to_array_udf = udf(vector_to_array, ArrayType(DoubleType()))
        
        # Convert feature vectors to arrays in both DataFrames
        train_df_arrays = train_df.withColumn("features_array", vector_to_array_udf(col("features")))
        test_df_arrays = test_df.withColumn("features_array", vector_to_array_udf(col("features")))
        
        # Broadcast model and scaler for distributed inference
        model_broadcast = self.spark.sparkContext.broadcast(self.sklearn_model)
        scaler_broadcast = self.spark.sparkContext.broadcast(self.feature_scaler)
        
        # Optimized pandas UDF with batch processing
        batch_size = self.mode_config['hybrid_config']['distributed_inference'].get('batch_size', 10000)
        
        @pandas_udf(returnType=DoubleType())
        def predict_udf_optimized(features_arrays):
            """Memory-optimized pandas UDF for distributed sklearn predictions"""
            import pandas as pd
            import numpy as np
            import gc
            
            # Get broadcasted objects
            model = model_broadcast.value
            scaler = scaler_broadcast.value
            
            # Convert pandas Series of arrays to numpy matrix
            features_matrix = np.array(features_arrays.tolist())
            
            # Process in batches to reduce memory pressure
            predictions = []
            batch_size_local = min(batch_size, len(features_matrix))
            
            for i in range(0, len(features_matrix), batch_size_local):
                end_idx = min(i + batch_size_local, len(features_matrix))
                batch_features = features_matrix[i:end_idx]
                
                # Apply scaling
                batch_features_scaled = scaler.transform(batch_features)
                
                # Make predictions
                batch_predictions = model.predict(batch_features_scaled)
                predictions.extend(batch_predictions)
                
                # Force garbage collection for large batches
                if len(batch_features) > 1000:
                    del batch_features_scaled
                    del batch_predictions
                    gc.collect()
            
            return pd.Series(predictions)
        
        # Optimize Spark DataFrames for distributed processing
        train_df_arrays = train_df_arrays.repartition(16)  # Optimize partitions
        test_df_arrays = test_df_arrays.repartition(16)
        
        # Apply distributed predictions using the features array column
        inference_start = time.time()
        
        self.logger.info(f"🚀 Starting distributed inference with batch size: {batch_size:,}")
        
        # Process train predictions with memory management
        train_predictions_df = train_df_arrays.withColumn("prediction", predict_udf_optimized(col("features_array")))
        
        # Cache and trigger computation for train set
        train_predictions_df.cache()
        train_count = train_predictions_df.count()
        
        # Process test predictions
        test_predictions_df = test_df_arrays.withColumn("prediction", predict_udf_optimized(col("features_array")))
        
        # Cache and trigger computation for test set
        test_predictions_df.cache()
        test_count = test_predictions_df.count()
        
        inference_time = time.time() - inference_start
        
        # Store distributed inference metrics with optimization info
        self.training_metrics['distributed_inference_metrics'] = {
            'inference_time': inference_time,
            'train_predictions_count': train_count,
            'test_predictions_count': test_count,
            'predictions_per_second': (train_count + test_count) / inference_time if inference_time > 0 else 0,
            'pandas_udf_used': True,
            'distributed_execution': True,
            'batch_processing': True,
            'batch_size': batch_size,
            'memory_optimized': True,
            'partitions_used': 16
        }
        
        self.logger.info(f"✅ Distributed inference completed in {inference_time:.2f}s")
        self.logger.info(f"📈 Inference throughput: {(train_count + test_count) / inference_time:.1f} predictions/sec")
        
        return train_predictions_df, test_predictions_df
    
    def _evaluate_hybrid_model(self, train_predictions_df: DataFrame, test_predictions_df: DataFrame,
                              feature_names: List[str]) -> Dict[str, Any]:
        """Evaluate hybrid model using distributed predictions"""
        
        self.logger.info("📊 Evaluating hybrid model performance...")
        
        from pyspark.ml.evaluation import RegressionEvaluator
        
        # Create evaluator
        evaluator = RegressionEvaluator(labelCol="target", predictionCol="prediction")
        
        # Calculate metrics for training set
        train_mae = evaluator.evaluate(train_predictions_df, {evaluator.metricName: "mae"})
        train_mse = evaluator.evaluate(train_predictions_df, {evaluator.metricName: "mse"})
        train_rmse = evaluator.evaluate(train_predictions_df, {evaluator.metricName: "rmse"})
        train_r2 = evaluator.evaluate(train_predictions_df, {evaluator.metricName: "r2"})
        
        # Calculate metrics for test set
        test_mae = evaluator.evaluate(test_predictions_df, {evaluator.metricName: "mae"})
        test_mse = evaluator.evaluate(test_predictions_df, {evaluator.metricName: "mse"})
        test_rmse = evaluator.evaluate(test_predictions_df, {evaluator.metricName: "rmse"})
        test_r2 = evaluator.evaluate(test_predictions_df, {evaluator.metricName: "r2"})
        
        # Get feature importance from sklearn model
        if hasattr(self.sklearn_model, 'feature_importances_'):
            feature_importance_array = self.sklearn_model.feature_importances_
        elif hasattr(self.sklearn_model, 'coef_'):
            feature_importance_array = np.abs(self.sklearn_model.coef_)
        else:
            feature_importance_array = np.ones(len(feature_names)) / len(feature_names)
        
        feature_importance_df = self.spark.createDataFrame(
            [(feature_names[i], float(feature_importance_array[i])) for i in range(len(feature_names))],
            ["feature", "importance"]
        ).orderBy("importance", ascending=False)
        
        evaluation_results = {
            'train_metrics': {
                'mae': train_mae,
                'mse': train_mse,
                'rmse': train_rmse,
                'r2': train_r2
            },
            'test_metrics': {
                'mae': test_mae,
                'mse': test_mse,
                'rmse': test_rmse,
                'r2': test_r2
            },
            'feature_importance': feature_importance_df,
            'train_predictions': train_predictions_df,
            'test_predictions': test_predictions_df
        }
        
        # Create visualizations (same types for comparison across modes)
        try:
            self._create_hybrid_visualizations(evaluation_results, feature_names)
            self.logger.info("✅ Hybrid-specific visualizations created")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create visualizations: {e}")
        
        return evaluation_results
    
    def _create_hybrid_visualizations(self, evaluation_results: Dict[str, Any], feature_names: List[str]):
        """Create hybrid-specific visualizations (same types for comparison)"""
        
        self.logger.info("📊 Creating hybrid sklearn visualizations...")
        
        try:
            # Convert Spark DataFrames to pandas for visualization compatibility
            
            # 1. Feature importance visualization
            feature_importance_spark_df = evaluation_results['feature_importance']
            feature_importance_pandas_df = feature_importance_spark_df.toPandas()
            self.visualizer.plot_feature_importance(feature_importance_pandas_df, top_n=20)
            
            # 2. Predictions vs actual visualization
            test_predictions_spark_df = evaluation_results['test_predictions']
            # Convert to list of row objects for the visualizer
            test_results_list = test_predictions_spark_df.select("stock_symbol", "target", "prediction", "Date").collect()
            self.visualizer.plot_predictions_vs_actual(test_results_list, sample_size=1000)
            
            # 3. Residuals analysis
            self.visualizer.plot_residuals(test_results_list)
            
            # Hybrid-specific: Training approach analysis
            self._create_hybrid_specific_plots()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create hybrid visualizations: {e}")
            # Create at least the hybrid-specific plots
            try:
                self._create_hybrid_specific_plots()
            except Exception as e2:
                self.logger.warning(f"⚠️ Failed to create hybrid-specific plots: {e2}")
    
    def _create_hybrid_specific_plots(self):
        """Create hybrid approach specific analysis plots"""
        
        import matplotlib.pyplot as plt
        
        # Hybrid-specific plot: Phase timing breakdown
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        phases = ['Data Loading', 'Collection', 'Sklearn Training', 'Distributed Inference', 'Evaluation']
        times = [
            self.training_metrics['data_loading_time'],
            self.training_metrics['data_collection_time'], 
            self.training_metrics['sklearn_training_time'],
            self.training_metrics['distributed_inference_time'],
            self.training_metrics['evaluation_time']
        ]
        
        ax.bar(phases, times)
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Hybrid Approach - Phase Timing Breakdown')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save to plots directory
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        plot_path = plots_dir / "hybrid_phase_timing_breakdown.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        # Note: Don't log to MLflow here as this will be logged to scalability monitoring experiment
        # by the MultiModeTrainer's monitoring system
            
        plt.close()
        
        self.logger.info("📊 Hybrid-specific plots created")
    
    def _collect_hybrid_scalability_metrics(self, total_samples: int, total_training_time: float) -> Dict:
        """Collect scalability-specific metrics for hybrid approach"""
        
        import psutil
        
        scalability_metrics = {
            'mode_specific_focus': 'driver_training_distributed_inference',
            'throughput_samples_per_second': total_samples / total_training_time if total_training_time > 0 else 0,
            'hybrid_characteristics': {
                'data_collection_overhead': self.training_metrics['data_collection_time'],
                'driver_training_time': self.training_metrics['sklearn_training_time'],
                'distributed_inference_time': self.training_metrics['distributed_inference_time'],
                'inference_throughput': self.training_metrics['distributed_inference_metrics'].get('predictions_per_second', 0)
            },
            'training_efficiency_score': self._calculate_hybrid_efficiency_score(total_samples, total_training_time),
            'hybrid_approach_characteristics': {
                'benefits': 'Sklearn ecosystem, fast training, distributed inference scalability',
                'tradeoffs': 'Data collection overhead, driver memory constraints',
                'optimal_use_case': 'Medium datasets with complex sklearn algorithms and distributed inference needs'
            },
            'resource_utilization': {
                'driver_memory_usage_mb': psutil.virtual_memory().used / (1024 * 1024),
                'data_transfer_efficiency': self._calculate_data_transfer_efficiency(),
                'inference_distribution_efficiency': self._calculate_inference_distribution_efficiency()
            }
        }
        
        return scalability_metrics
    
    def _calculate_hybrid_efficiency_score(self, total_samples: int, total_training_time: float) -> float:
        """Calculate hybrid approach specific efficiency score"""
        
        if total_samples == 0 or total_training_time == 0:
            return 0.0
        
        # Expected baseline for hybrid: ~1500 samples per second (considering overhead)
        baseline_throughput = 1500.0
        actual_throughput = total_samples / total_training_time
        
        efficiency_score = min(1.0, actual_throughput / baseline_throughput)
        
        return efficiency_score
    
    def _calculate_data_transfer_efficiency(self) -> float:
        """Calculate efficiency of data collection to driver"""
        
        collection_time = self.training_metrics['data_collection_time']
        total_samples = self.training_metrics['total_samples']
        
        if collection_time == 0 or total_samples == 0:
            return 0.0
        
        # Target: collect 10000 samples per second
        target_rate = 10000.0
        actual_rate = total_samples / collection_time
        
        efficiency = min(1.0, actual_rate / target_rate)
        return efficiency
    
    def _calculate_inference_distribution_efficiency(self) -> float:
        """Calculate efficiency of distributed inference"""
        
        inference_metrics = self.training_metrics['distributed_inference_metrics']
        predictions_per_second = inference_metrics.get('predictions_per_second', 0)
        
        if predictions_per_second == 0:
            return 0.0
        
        # Target: 5000 predictions per second for distributed inference
        target_rate = 5000.0
        
        efficiency = min(1.0, predictions_per_second / target_rate)
        return efficiency
    
    def get_training_summary(self) -> Dict:
        """Get training summary for reporting"""
        
        return {
            'mode': 'hybrid_sklearn',
            'description': 'Hybrid approach - Spark preprocessing + sklearn training + distributed inference',
            'characteristics': {
                'preprocessing': 'distributed_spark_preprocessing',
                'training_type': 'sklearn_on_driver',
                'inference': 'distributed_pandas_udf',
                'memory_focus': 'driver_collection_and_training',
                'scalability_pattern': 'moderate_data_transfer_overhead'
            },
            'monitoring_focus': [
                'data_transfer_overhead',
                'driver_sklearn_training',
                'distributed_inference_performance',
                'memory_driver_vs_executor_balance'
            ],
            'metrics': self.training_metrics
        }