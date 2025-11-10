#!/usr/bin/env python3
"""
Enhanced Inference Script with Scalability Monitoring
Provides production-ready inference with comprehensive performance tracking

CONDA ENVIRONMENT: breaking_data

Usage: 
    # Run inference for all stocks with monitoring (default)
    conda activate breaking_data
    python scripts/inference_with_monitoring.py

    # Run inference for specific stocks with monitoring
    python scripts/inference_with_monitoring.py --stocks AAPL,GOOG,NVDA
    
    # Disable monitoring
    python scripts/inference_with_monitoring.py --no-monitoring
    
    # Run concurrent load test
    python scripts/inference_with_monitoring.py --load-test
    
Features:
- Real-time scalability monitoring
- Performance bottleneck detection
- Resource utilization tracking
- Concurrent inference testing
- MLflow experiment tracking
- Automated report generation
"""

import sys
import os
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Add src to path
pipeline_root = Path(__file__).parent.parent
sys.path.insert(0, str(pipeline_root))

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, max as spark_max, lit, when
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StandardScaler
from delta import configure_spark_with_delta_pip
import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient

# Local imports
from src.utils import Config, Logger, MLflowManager, PathManager, load_environment
from src.data_processing import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.scalability_monitor import ScalabilityMonitor
from src.monitoring_config_manager import get_monitoring_config_manager


class MonitoredInferenceEngine:
    """
    Production inference engine with comprehensive scalability monitoring
    Supports both unified and per-stock model architectures
    """
    
    def __init__(self, config: Config, spark: SparkSession, enable_monitoring: bool = True):
        self.config = config
        self.spark = spark
        self.logger = Logger().get_logger()
        
        # Initialize core components
        self.data_processor = DataProcessor(config, spark)
        self.feature_engineer = FeatureEngineer(config, spark)
        self.model_trainer = ModelTrainer(config, spark)
        self.mlflow_manager = MLflowManager(config)
        
        # Initialize monitoring
        self.enable_monitoring = enable_monitoring and config.monitoring.enabled
        if self.enable_monitoring:
            monitoring_config = {
                'enabled': True,
                'detailed_metrics': config.monitoring.detailed_metrics,
                'resource_monitoring_interval': config.monitoring.resource_monitoring_interval,
                'experiment_name': config.monitoring.experiment_name + "_inference"
            }
            self.monitor = ScalabilityMonitor(config, spark, monitoring_config)
            self.logger.info("🔍 Inference monitoring enabled")
        else:
            self.monitor = None
            self.logger.info("⚠️  Inference monitoring disabled")
        
        # Inference settings from config - LOAD MODELS THROUGH CONFIG ONLY
        self.lookback_days = config.inference.lookback_days
        self.model_name = config.inference.model_name
        self.model_stage = config.inference.model_stage
        self.model_version = config.inference.model_version
        
        # Initialize model architecture detection
        self.is_unified_model = False
        self.unified_stocks = []
        
        # Load model and scaler on initialization
        self.model = None
        self.feature_scaler = None
        self.feature_names = None
        self._load_unified_model()
        self._load_feature_scaler()
        
        # Inference metrics
        self.inference_metrics = {
            'start_time': None,
            'end_time': None,
            'total_inference_time': 0,
            'stocks_processed': [],
            'predictions': {},
            'errors': [],
            'performance_metrics': {},
            'scalability_metrics': None,
            'model_type': 'unified' if self.is_unified_model else 'per_stock',
            'model_name_used': self.model_name
        }
    
    def _load_unified_model(self) -> None:
        """Load unified model that can predict for multiple stocks - from config only"""
        model_load_start = time.time()
        
        try:
            self.logger.info(f"Loading unified model {self.model_name} from MLflow...")
            
            # Set MLflow tracking URI
            mlflow.set_tracking_uri(str(self.config.paths.mlflow_tracking_dir))
            
            # Load model based on version specification from config
            if self.model_version == "latest":
                client = MlflowClient()
                try:
                    model_versions = client.get_latest_versions(
                        self.model_name, 
                        stages=[self.model_stage] if self.model_stage != "None" else None
                    )
                    if not model_versions:
                        # Try without stage filter
                        all_versions = client.search_model_versions(f"name='{self.model_name}'")
                        if all_versions:
                            # Get the latest version
                            latest_version = max(all_versions, key=lambda v: int(v.version))
                            model_uri = f"models:/{self.model_name}/{latest_version.version}"
                        else:
                            raise ValueError(f"No versions found for model {self.model_name}")
                    else:
                        model_version = model_versions[0]
                        model_uri = f"models:/{self.model_name}/{model_version.version}"
                except Exception as e:
                    self.logger.warning(f"Failed to get latest version: {e}. Trying latest version number...")
                    # Fallback: get all versions and pick the highest number
                    all_versions = client.search_model_versions(f"name='{self.model_name}'")
                    if all_versions:
                        latest_version = max(all_versions, key=lambda v: int(v.version))
                        model_uri = f"models:/{self.model_name}/{latest_version.version}"
                        self.logger.info(f"Using model version {latest_version.version}")
                    else:
                        raise ValueError(f"Model {self.model_name} not found in MLflow")
            else:
                model_uri = f"models:/{self.model_name}/{self.model_version}"
            
            self.logger.info(f"Loading unified model from URI: {model_uri}")
            self.model = mlflow.spark.load_model(model_uri)
            
            # Detect if this is a unified model by checking the model name
            self._detect_model_architecture()
            
            # Extract feature names from model metadata
            self._extract_feature_names(model_uri)
            
            # Monitor model loading performance
            if self.monitor:
                self.monitor.monitor_performance("model_loading", model_load_start, 1)
            
            model_type = "unified" if self.is_unified_model else "per-stock"
            self.logger.info(f"✅ {model_type} model loaded successfully")
            
            if self.is_unified_model:
                self.logger.info(f"🔗 Unified model supports stocks: {self.unified_stocks}")
            
        except Exception as e:
            self.logger.error(f"Failed to load unified model: {str(e)}")
            raise RuntimeError(f"Unified model loading failed: {str(e)}")
    
    def _detect_model_architecture(self) -> None:
        """Detect if loaded model is unified (supports multiple stocks) or per-stock"""
        model_name_lower = self.model_name.lower()
        
        # Check if model name indicates unified architecture
        if any(keyword in model_name_lower for keyword in ['unified', 'multi', 'combined']):
            self.is_unified_model = True
            
            # Extract stock information from model name
            if 'unified' in model_name_lower:
                # Extract number or list of stocks from name like "stock_predictor_unified_5stocks"
                if '5stocks' in model_name_lower:
                    self.unified_stocks = self.config.data.training_stocks[:5]
                elif '4stocks' in model_name_lower:
                    self.unified_stocks = self.config.data.training_stocks[:4]
                else:
                    # Default to all available training stocks
                    self.unified_stocks = self.config.data.training_stocks
            elif any(stock in model_name_lower for stock in self.config.data.available_stocks):
                # Extract specific stocks from model name like "stock_predictor_AAPL_GOOG"
                found_stocks = [stock for stock in self.config.data.available_stocks 
                              if stock.lower() in model_name_lower]
                self.unified_stocks = found_stocks
                self.is_unified_model = len(found_stocks) > 1
            else:
                # Default unified model
                self.unified_stocks = self.config.data.training_stocks
                
        else:
            # Per-stock model
            self.is_unified_model = False
            self.unified_stocks = []
            
        self.logger.info(f"Model architecture detected: {'Unified' if self.is_unified_model else 'Per-stock'}")
        if self.is_unified_model:
            self.logger.info(f"Unified model stocks: {self.unified_stocks}")
    
    def _extract_feature_names(self, model_uri: str) -> None:
        """Extract feature names using FeatureEngineer"""
        try:
            self.logger.info("Extracting feature names...")
            
            # Get feature names by creating features from a sample stock
            sample_stock = self.config.data.inference_stocks[0]
            stock_table = f"stock_{sample_stock}"
            
            # Load a small sample of data
            sample_df = self.spark.read.format("delta").load(f"delta_tables/{stock_table}").limit(50)
            
            # Create features using FeatureEngineer (same as training)
            featured_df, feature_names = self.feature_engineer.create_all_features(sample_df)
            
            self.feature_names = feature_names
            
            self.logger.info(f"Extracted {len(self.feature_names)} feature names")
                
        except Exception as e:
            self.logger.warning(f"Could not extract feature names: {e}")
            # Final fallback
            self.feature_names = [
                'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_5',
                'price_change_1d', 'price_change_3d', 'price_change_5d',
                'ma_5', 'ma_10', 'ma_20', 'close_vs_ma5', 'close_vs_ma10', 'close_vs_ma20'
            ]
    
    def _load_feature_scaler(self) -> None:
        """Load the feature scaler used during training"""
        try:
            from pyspark.ml.feature import StandardScalerModel
            
            scaler_path = "models/feature_scaler"
            self.logger.info(f"Loading feature scaler from {scaler_path}...")
            
            self.feature_scaler = StandardScalerModel.load(scaler_path)
            
            self.logger.info("✅ Feature scaler loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load feature scaler: {str(e)}")
            self.feature_scaler = None
            raise
    
    def run_monitored_inference(self, stocks: Optional[List[str]] = None, run_load_test: bool = False) -> Dict:
        """
        Run inference with comprehensive monitoring
        Uses unified model architecture if available, falls back to per-stock processing
        
        Args:
            stocks: List of stock symbols. If None, uses config.data.inference_stocks
            run_load_test: Whether to run concurrent load testing
            
        Returns:
            Dictionary with inference results and scalability metrics
        """
        self.inference_metrics['start_time'] = datetime.now()
        
        # Start monitoring
        if self.monitor:
            self.monitor.start_monitoring("inference")
        
        # Use config stocks if none specified
        if stocks is None:
            stocks = self.config.data.inference_stocks
        
        self.logger.info(f"🔮 Starting monitored inference for stocks: {stocks}")
        self.logger.info(f"📦 Using {self.inference_metrics['model_type']} model: {self.model_name}")
        
        try:
            # Monitor data volume before inference
            if self.monitor:
                data_volume_metrics = self.monitor.monitor_data_volume("inference_data_analysis")
                self.inference_metrics['data_volume_metrics'] = data_volume_metrics
                
                # Check data volume warnings
                self._check_inference_data_warnings(data_volume_metrics)
            
            # Run load testing if requested
            if run_load_test:
                self.logger.info("🏃 Running concurrent inference load test")
                load_test_results = self._run_concurrent_load_test(stocks)
                self.inference_metrics['load_test_results'] = load_test_results
            
            # Choose inference method based on model architecture
            if self.is_unified_model:
                self.logger.info("🔗 Using unified inference pipeline")
                inference_results = self._run_unified_inference_with_monitoring(stocks)
            else:
                self.logger.info("🔄 Using per-stock inference pipeline")
                inference_results = self._run_inference_with_monitoring(stocks)
                
            self.inference_metrics.update(inference_results)
            
            # Calculate inference efficiency
            if self.monitor:
                self._calculate_inference_efficiency()
                
        except Exception as e:
            self.logger.error(f"Monitored inference failed: {e}")
            self.inference_metrics['error'] = str(e)
            raise
            
        finally:
            # Stop monitoring and collect results
            if self.monitor:
                scalability_metrics = self.monitor.stop_monitoring()
                self.inference_metrics['scalability_metrics'] = scalability_metrics
                
                # Update monitoring config with latest run info
                try:
                    config_manager = get_monitoring_config_manager()
                    if hasattr(self.monitor, 'current_run_id') and self.monitor.current_run_id:
                        experiment_name = self.config.monitoring.experiment_name + "_inference"
                        config_manager.update_inference_monitoring_config(
                            self.monitor.current_run_id, 
                            experiment_name
                        )
                        self.logger.info(f"✅ Updated inference monitoring config - Run ID: {self.monitor.current_run_id}")
                        self.logger.info(f"   📊 Experiment: {experiment_name}")
                        
                        # Add run info to inference results
                        self.inference_metrics['mlflow_run_id'] = self.monitor.current_run_id
                        self.inference_metrics['mlflow_experiment_name'] = experiment_name
                    else:
                        self.logger.warning("⚠️ No MLflow run ID available from monitor for config update")
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to update inference monitoring config: {e}")
                
                # Generate monitoring report
                self._generate_inference_report(scalability_metrics)
            
            self.inference_metrics['end_time'] = datetime.now()
            self.inference_metrics['total_inference_time'] = (
                self.inference_metrics['end_time'] - self.inference_metrics['start_time']
            ).total_seconds()
        
        # Log summary
        self._log_inference_summary()
        
        return self.inference_metrics
    
    def _run_unified_inference_with_monitoring(self, stocks: List[str]) -> Dict:
        """
        Run unified inference that processes multiple stocks with one model
        Creates features per stock separately, then makes predictions with unified model
        """
        self.logger.info(f"🔗 Starting unified inference for {len(stocks)} stocks")
        
        results = {
            'stocks_processed': [],
            'predictions': {},
            'errors': [],
            'performance_metrics': {
                'total_predictions': 0,
                'successful_predictions': 0,
                'failed_predictions': 0,
                'per_stock_metrics': {},
                'unified_processing_time': 0
            }
        }
        
        unified_start_time = time.time()
        
        # Validate stocks are supported by unified model
        unsupported_stocks = [s for s in stocks if s not in self.unified_stocks]
        if unsupported_stocks:
            self.logger.warning(f"⚠️  Stocks not supported by unified model: {unsupported_stocks}")
            self.logger.info(f"Unified model supports: {self.unified_stocks}")
            # Filter to only supported stocks
            stocks = [s for s in stocks if s in self.unified_stocks]
            if not stocks:
                raise ValueError("No supported stocks found for unified model")
        
        # Process each stock separately for feature creation (like in inference.py)
        stock_features = {}
        
        for stock in stocks:
            try:
                self.logger.info(f"📊 Creating features for {stock}...")
                
                stock_start_time = time.time()
                
                # Load recent data for this stock
                recent_data = self._load_recent_data(stock)
                
                # Validate data quality
                self._validate_data_quality(recent_data, stock)
                
                # Prepare features for this specific stock (like inference.py approach)
                feature_data = self._prepare_features_for_stock(recent_data, stock)
                
                # Store prepared features for unified prediction
                stock_features[stock] = feature_data
                
                stock_duration = time.time() - stock_start_time
                
                # Monitor per-stock feature preparation
                if self.monitor:
                    stock_performance = self.monitor.monitor_performance(
                        f"feature_prep_{stock}", stock_start_time, 1
                    )
                    results['performance_metrics']['per_stock_metrics'][stock] = stock_performance
                
                self.logger.info(f"✅ Features created for {stock} in {stock_duration:.2f}s")
                
            except Exception as e:
                error_msg = f"Feature preparation failed for {stock}: {str(e)}"
                self.logger.error(error_msg)
                results['errors'].append(error_msg)
                continue
        
        # Make unified predictions for all stocks with valid features
        if stock_features:
            try:
                self.logger.info(f"🎯 Making unified predictions for {len(stock_features)} stocks")
                
                prediction_start_time = time.time()
                
                # Make predictions for each stock using the unified model
                for stock, feature_data in stock_features.items():
                    try:
                        # Make prediction using unified model
                        prediction = self._make_prediction(feature_data, stock)
                        
                        # Create prediction result for this stock
                        recent_data = self._load_recent_data(stock)  # Reload for metadata
                        prediction_result = self._create_prediction_result(
                            stock, prediction, recent_data
                        )
                        
                        results['stocks_processed'].append(stock)
                        results['predictions'][stock] = prediction_result
                        results['performance_metrics']['successful_predictions'] += 1
                        
                        # Save prediction if configured
                        if self.config.inference.save_predictions:
                            self._save_prediction(prediction_result)
                            
                    except Exception as e:
                        error_msg = f"Unified prediction failed for {stock}: {str(e)}"
                        self.logger.error(error_msg)
                        results['errors'].append(error_msg)
                        results['performance_metrics']['failed_predictions'] += 1
                
                prediction_duration = time.time() - prediction_start_time
                
                # Monitor unified prediction performance
                if self.monitor:
                    self.monitor.monitor_performance(
                        "unified_prediction", prediction_start_time, len(stock_features)
                    )
                
                self.logger.info(f"✅ Unified predictions completed in {prediction_duration:.2f}s")
                
            except Exception as e:
                self.logger.error(f"Unified prediction pipeline failed: {e}")
                results['errors'].append(f"Unified prediction error: {str(e)}")
        
        # Calculate total metrics
        results['performance_metrics']['total_predictions'] = (
            results['performance_metrics']['successful_predictions'] + 
            results['performance_metrics']['failed_predictions']
        )
        results['performance_metrics']['unified_processing_time'] = time.time() - unified_start_time
        
        self.logger.info(
            f"🎯 Unified inference completed: {results['performance_metrics']['successful_predictions']}"
            f"/{results['performance_metrics']['total_predictions']} successful"
        )
        
        return results
    
    def _run_inference_with_monitoring(self, stocks: List[str]) -> Dict:
        """Run inference for stocks with detailed performance monitoring"""
        
        results = {
            'stocks_processed': [],
            'predictions': {},
            'errors': [],
            'performance_metrics': {
                'total_predictions': 0,
                'successful_predictions': 0,
                'failed_predictions': 0,
                'per_stock_metrics': {}
            }
        }
        
        # Process each stock with monitoring
        for stock in stocks:
            try:
                self.logger.info(f"📊 Processing inference for {stock}...")
                
                stock_start_time = time.time()
                prediction_result = self._run_single_stock_inference_monitored(stock)
                stock_duration = time.time() - stock_start_time
                
                results['stocks_processed'].append(stock)
                results['predictions'][stock] = prediction_result
                results['performance_metrics']['total_predictions'] += 1
                results['performance_metrics']['successful_predictions'] += 1
                
                # Store per-stock performance metrics
                if self.monitor:
                    stock_performance = self.monitor.monitor_performance(
                        f"inference_{stock}", stock_start_time, 1
                    )
                    results['performance_metrics']['per_stock_metrics'][stock] = stock_performance
                
                self.logger.info(f"✅ {stock} inference completed in {stock_duration:.2f}s")
                
            except Exception as e:
                error_msg = f"Inference failed for {stock}: {str(e)}"
                self.logger.error(error_msg)
                results['errors'].append(error_msg)
                results['performance_metrics']['failed_predictions'] += 1
        
        return results
    
    def _run_single_stock_inference_monitored(self, stock: str) -> Dict:
        """Run inference for a single stock with detailed monitoring"""
        
        # Monitor data loading
        data_load_start = time.time()
        recent_data = self._load_recent_data(stock)
        
        if self.monitor:
            self.monitor.monitor_performance("data_loading", data_load_start, recent_data.count())
        
        # Validate data quality
        self._validate_data_quality(recent_data, stock)
        
        # Monitor feature engineering
        feature_start = time.time()
        feature_data = self._prepare_features(recent_data, stock)
        
        if self.monitor:
            self.monitor.monitor_performance("feature_engineering", feature_start, len(self.feature_names))
        
        # Monitor model prediction
        prediction_start = time.time()
        prediction = self._make_prediction(feature_data, stock)
        
        if self.monitor:
            self.monitor.monitor_performance("model_prediction", prediction_start, 1)
        
        # Create per-stock prediction result
        prediction_result = self._create_prediction_result(stock, prediction, recent_data)
        
        # Override model type for per-stock approach
        prediction_result['model_type'] = 'per_stock'
        
        # Save prediction if configured
        if self.config.inference.save_predictions:
            self._save_prediction(prediction_result)
        
        return prediction_result
    
    def _run_concurrent_load_test(self, stocks: List[str]) -> Dict:
        """Run concurrent inference load test"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        load_test_results = {
            'concurrent_tests': [],
            'performance_summary': {}
        }
        
        # Test different concurrency levels
        concurrency_levels = self.config.monitoring.load_testing.concurrent_users
        
        for concurrency in concurrency_levels:
            self.logger.info(f"🏃 Testing {concurrency} concurrent inference requests")
            
            start_time = time.time()
            successful_requests = 0
            failed_requests = 0
            response_times = []
            
            # Create thread pool
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                # Submit concurrent inference tasks
                futures = []
                for i in range(concurrency):
                    stock = stocks[i % len(stocks)]  # Cycle through available stocks
                    future = executor.submit(self._single_threaded_inference, stock, i)
                    futures.append(future)
                
                # Collect results
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=30)  # 30 second timeout
                        successful_requests += 1
                        response_times.append(result['response_time'])
                    except Exception as e:
                        failed_requests += 1
                        self.logger.warning(f"Concurrent request failed: {e}")
            
            total_time = time.time() - start_time
            
            # Calculate statistics
            if response_times:
                avg_response_time = np.mean(response_times)
                p95_response_time = np.percentile(response_times, 95)
                throughput = successful_requests / total_time
            else:
                avg_response_time = 0
                p95_response_time = 0
                throughput = 0
            
            test_result = {
                'concurrency_level': concurrency,
                'total_time': total_time,
                'successful_requests': successful_requests,
                'failed_requests': failed_requests,
                'avg_response_time': avg_response_time,
                'p95_response_time': p95_response_time,
                'throughput': throughput,
                'success_rate': successful_requests / (successful_requests + failed_requests) if (successful_requests + failed_requests) > 0 else 0
            }
            
            load_test_results['concurrent_tests'].append(test_result)
            
            self.logger.info(
                f"Concurrency {concurrency}: {throughput:.1f} req/sec, "
                f"{avg_response_time:.2f}s avg, {test_result['success_rate']:.1%} success"
            )
        
        return load_test_results
    
    def _single_threaded_inference(self, stock: str, request_id: int) -> Dict:
        """Single threaded inference for load testing"""
        request_start = time.time()
        
        try:
            # Simplified inference for load testing
            recent_data = self._load_recent_data(stock)
            feature_data = self._prepare_features(recent_data, stock)
            prediction = self._make_prediction(feature_data, stock)
            
            response_time = time.time() - request_start
            
            return {
                'request_id': request_id,
                'stock': stock,
                'prediction': float(prediction),
                'response_time': response_time,
                'success': True
            }
            
        except Exception as e:
            response_time = time.time() - request_start
            return {
                'request_id': request_id,
                'stock': stock,
                'error': str(e),
                'response_time': response_time,
                'success': False
            }
    
    def _check_inference_data_warnings(self, data_volume_metrics: Dict):
        """Check and warn about inference data issues"""
        total_rows = data_volume_metrics.get('total_rows', 0)
        
        # Check for insufficient data
        min_required = self.lookback_days * len(self.config.data.inference_stocks)
        if total_rows < min_required:
            self.logger.warning(
                f"⚠️  Limited data for inference: {total_rows:,} rows. "
                f"Recommended: {min_required:,} rows for optimal predictions."
            )
    
    def _calculate_inference_efficiency(self):
        """Calculate inference efficiency metrics with improved logic"""
        if not self.monitor:
            self.logger.info("⚠️  No monitor available - using fallback efficiency calculations")
            
        # Calculate efficiency scores with improved error handling
        inference_efficiency = {
            'latency_efficiency': self._calculate_latency_efficiency(),
            'throughput_efficiency': self._calculate_throughput_efficiency(),
            'resource_efficiency': self._calculate_resource_efficiency(),
            'prediction_quality_efficiency': self._calculate_prediction_quality_efficiency()
        }
        
        self.inference_metrics['inference_efficiency'] = inference_efficiency
        
        # Log efficiency details for debugging
        self.logger.info("📊 Efficiency Calculation Details:")
        for metric, value in inference_efficiency.items():
            self.logger.info(f"   {metric}: {value:.3f}")
        
        # Log efficiency warnings with more reasonable thresholds
        for metric, value in inference_efficiency.items():
            if value < 0.3:  # Lower threshold - below 30% efficiency
                self.logger.warning(f"⚠️  Low {metric}: {value:.3f}")
            elif value < 0.5:  # Medium threshold
                self.logger.info(f"🟡 Moderate {metric}: {value:.3f}")
            else:
                self.logger.info(f"✅ Good {metric}: {value:.3f}")
        
        # Add monitoring data debug info
        if self.monitor and hasattr(self.monitor, 'metrics') and self.monitor.metrics:
            metrics = self.monitor.metrics
            self.logger.debug("🔍 Monitor metrics available:")
            self.logger.debug(f"   avg_processing_time_per_record: {getattr(metrics, 'avg_processing_time_per_record', 'N/A')}")
            self.logger.debug(f"   throughput_records_per_second: {getattr(metrics, 'throughput_records_per_second', 'N/A')}")
            self.logger.debug(f"   avg_memory_usage_mb: {getattr(metrics, 'avg_memory_usage_mb', 'N/A')}")
            self.logger.debug(f"   avg_cpu_usage_percent: {getattr(metrics, 'avg_cpu_usage_percent', 'N/A')}")
        else:
            self.logger.debug("🔍 No monitor metrics available - using fallback calculations")
    
    def _calculate_latency_efficiency(self) -> float:
        """Calculate latency efficiency score for inference workload"""
        if not self.monitor or not hasattr(self.monitor, 'metrics') or not self.monitor.metrics:
            return 0.5  # Neutral score if no monitoring data
            
        # For inference workload: Target < 5 seconds per prediction (more realistic)
        target_latency_ms = 5000  # 5 seconds
        actual_latency_ms = getattr(self.monitor.metrics, 'avg_processing_time_per_record', 0)
        
        if actual_latency_ms <= 0:
            # Fallback: calculate from total time and successful predictions
            total_time_ms = self.inference_metrics.get('total_inference_time', 0) * 1000
            successful_predictions = len(self.inference_metrics.get('predictions', {}))
            if successful_predictions > 0:
                actual_latency_ms = total_time_ms / successful_predictions
            else:
                return 0.5  # Neutral score
        
        # Calculate efficiency score (1.0 = meeting target, > 1.0 = better than target)
        if actual_latency_ms <= target_latency_ms:
            return min(1.0, target_latency_ms / max(actual_latency_ms, 100))  # Cap very fast responses
        else:
            # Graceful degradation for slower responses
            return max(0.1, target_latency_ms / actual_latency_ms)
    
    def _calculate_throughput_efficiency(self) -> float:
        """Calculate throughput efficiency score for inference workload"""
        if not self.monitor or not hasattr(self.monitor, 'metrics') or not self.monitor.metrics:
            return 0.5  # Neutral score if no monitoring data
            
        # For inference workload: Target > 1 prediction per second (more realistic)
        target_throughput = 1.0  # 1 prediction/second
        actual_throughput = getattr(self.monitor.metrics, 'throughput_records_per_second', 0)
        
        if actual_throughput <= 0:
            # Fallback: calculate from metrics
            total_time = self.inference_metrics.get('total_inference_time', 0)
            successful_predictions = len(self.inference_metrics.get('predictions', {}))
            if total_time > 0 and successful_predictions > 0:
                actual_throughput = successful_predictions / total_time
            else:
                return 0.5  # Neutral score
        
        # Calculate efficiency score
        return min(1.0, actual_throughput / target_throughput)
    
    def _calculate_resource_efficiency(self) -> float:
        """Calculate resource efficiency score"""
        if not self.monitor or not hasattr(self.monitor, 'metrics') or not self.monitor.metrics:
            return 0.5  # Neutral score if no monitoring data
            
        # Check if we have resource metrics
        avg_memory_mb = getattr(self.monitor.metrics, 'avg_memory_usage_mb', 0)
        avg_cpu_percent = getattr(self.monitor.metrics, 'avg_cpu_usage_percent', 0)
        
        if not avg_memory_mb and not avg_cpu_percent:
            return 0.5  # Neutral score if no resource data
            
        # Calculate individual efficiency scores
        memory_efficiency = self._calculate_memory_utilization_score() if avg_memory_mb > 0 else 0.5
        cpu_efficiency = self._calculate_cpu_utilization_score() if avg_cpu_percent > 0 else 0.5
        
        return (memory_efficiency + cpu_efficiency) / 2
    
    def _calculate_memory_utilization_score(self) -> float:
        """Calculate memory utilization efficiency"""
        if not self.monitor or not hasattr(self.monitor, 'metrics') or not self.monitor.metrics:
            return 0.5
            
        usage_mb = getattr(self.monitor.metrics, 'avg_memory_usage_mb', 0)
        if usage_mb <= 0:
            return 0.5  # Neutral score if no data
            
        # Assume available system memory (from monitoring report: ~23GB peak usage)
        total_memory_mb = 32 * 1024  # 32GB assumption
        usage_percent = (usage_mb / total_memory_mb) * 100
        
        # Adjusted optimal ranges for inference workload:
        # - 20-80% utilization is good
        # - Below 10% is underutilization
        # - Above 90% is overutilization
        if usage_percent < 10:
            return usage_percent / 20  # Underutilization penalty
        elif usage_percent > 90:
            return max(0.1, (100 - usage_percent) / 10)  # Overutilization penalty
        else:
            # Good utilization range - scale to 0.6-1.0
            normalized = (usage_percent - 10) / 70  # 0-1 for 10-80%
            return 0.6 + (normalized * 0.4)  # Scale to 0.6-1.0
    
    def _calculate_cpu_utilization_score(self) -> float:
        """Calculate CPU utilization efficiency"""
        if not self.monitor or not hasattr(self.monitor, 'metrics') or not self.monitor.metrics:
            return 0.5
            
        usage_percent = getattr(self.monitor.metrics, 'avg_cpu_usage_percent', 0)
        if usage_percent <= 0:
            return 0.5  # Neutral score if no data
        
        # Adjusted optimal ranges for inference workload:
        # - 20-80% utilization is good
        # - Below 10% is underutilization  
        # - Above 90% is overutilization
        if usage_percent < 10:
            return usage_percent / 20  # Underutilization penalty
        elif usage_percent > 90:
            return max(0.1, (100 - usage_percent) / 10)  # Overutilization penalty
        else:
            # Good utilization range - scale to 0.6-1.0
            normalized = (usage_percent - 10) / 70  # 0-1 for 10-80%
            return 0.6 + (normalized * 0.4)  # Scale to 0.6-1.0
    
    def _calculate_prediction_quality_efficiency(self) -> float:
        """Calculate prediction quality efficiency (simplified)"""
        # This would typically involve comparing predictions with actual results
        # For now, use a simplified model based on prediction stability
        
        successful_predictions = len(self.inference_metrics.get('predictions', {}))
        total_attempts = successful_predictions + len(self.inference_metrics.get('errors', []))
        
        if total_attempts == 0:
            return 0.0
            
        success_rate = successful_predictions / total_attempts
        return success_rate
    
    def _calculate_prediction_confidence(self, prediction: float, stock: str) -> float:
        """Calculate confidence score for prediction"""
        # Simplified confidence based on prediction magnitude
        # In production, this would use model uncertainty estimates
        
        abs_prediction = abs(prediction)
        
        # Higher confidence for smaller predictions (more conservative)
        if abs_prediction < 0.05:  # < 5% change
            return 0.8
        elif abs_prediction < 0.1:  # < 10% change
            return 0.6
        elif abs_prediction < 0.2:  # < 20% change
            return 0.4
        else:
            return 0.2  # Low confidence for extreme predictions
    
    def _validate_prediction_reasonableness(self, predicted_gain: float, stock: str) -> bool:
        """Validate if prediction is within reasonable bounds"""
        # Check for extreme predictions (more than 50% change)
        if abs(predicted_gain) > 0.5:
            self.logger.warning(
                f"Extreme prediction for {stock}: {predicted_gain:.4f} "
                f"({predicted_gain*100:.2f}%) - rejecting"
            )
            return False
        
        # Check for very large predictions (more than 20% change)
        if abs(predicted_gain) > 0.2:
            self.logger.warning(
                f"Large prediction for {stock}: {predicted_gain:.4f} "
                f"({predicted_gain*100:.2f}%)"
            )
        
        return True
    
    def _generate_inference_report(self, scalability_metrics):
        """Generate comprehensive inference report"""
        report_path = f"results/inference/inference_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        # Generate monitoring report
        monitoring_report = self.monitor.generate_monitoring_report(report_path.replace('.txt', '_monitoring.txt'))
        
        # Generate inference-specific report
        inference_report = self._create_inference_report()
        
        # Combine reports
        combined_report = f"""
{inference_report}

{monitoring_report}
"""
        
        with open(report_path, 'w') as f:
            f.write(combined_report)
        
        self.logger.info(f"📄 Inference report saved to: {report_path}")
        
        # Export metrics to JSON
        metrics_path = report_path.replace('.txt', '_metrics.json')
        if self.monitor:
            self.monitor.export_metrics_json(metrics_path)
    
    def _create_inference_report(self) -> str:
        """Create inference-specific report"""
        report_lines = [
            "=" * 80,
            "🔮 INFERENCE PIPELINE REPORT", 
            "=" * 80,
            f"Inference Start: {self.inference_metrics['start_time']}",
            f"Inference End: {self.inference_metrics['end_time']}",
            f"Total Inference Time: {self.inference_metrics['total_inference_time']:.2f} seconds",
            "",
            "� MODEL CONFIGURATION:",
            "-" * 40,
            f"Model Architecture: {self.inference_metrics['model_type']}",
            f"Model Name: {self.inference_metrics['model_name_used']}",
            f"Model Version: {self.model_version}",
            f"Model Stage: {self.model_stage}",
            f"Lookback Days: {self.lookback_days}",
        ]
        
        if self.is_unified_model:
            report_lines.extend([
                f"Unified Model Stocks: {', '.join(self.unified_stocks)}",
            ])
        
        report_lines.extend([
            "",
            "�📊 INFERENCE RESULTS:",
            "-" * 40,
            f"Stocks Processed: {len(self.inference_metrics['stocks_processed'])}",
            f"Successful Predictions: {len(self.inference_metrics['predictions'])}",
            f"Failed Predictions: {len(self.inference_metrics['errors'])}",
        ])
        
        # Add individual predictions with per-stock details
        for stock, prediction in self.inference_metrics.get('predictions', {}).items():
            confidence = prediction.get('confidence_score', 0)
            model_type = prediction.get('model_type', 'unknown')
            report_lines.extend([
                f"",
                f"📈 {stock} ({model_type} model):",
                f"  Last Close: ${prediction['last_close']:.2f}",
                f"  Predicted Close: ${prediction['predicted_close']:.2f}",
                f"  Predicted Gain: {prediction['predicted_gain']:.4f} ({prediction['predicted_gain']*100:.2f}%)",
                f"  Confidence: {confidence:.2f}",
                f"  Next Trading Date: {prediction['next_trading_date'].strftime('%Y-%m-%d') if hasattr(prediction['next_trading_date'], 'strftime') else prediction['next_trading_date']}",
                f"  Rows Used: {prediction['rows_used']}",
            ])
        
        # Add load test results if available
        if 'load_test_results' in self.inference_metrics:
            report_lines.extend([
                "",
                "🏃 LOAD TEST RESULTS:",
                "-" * 40,
            ])
            
            for test in self.inference_metrics['load_test_results']['concurrent_tests']:
                report_lines.extend([
                    f"Concurrency {test['concurrency_level']}:",
                    f"  Throughput: {test['throughput']:.1f} req/sec",
                    f"  Avg Response Time: {test['avg_response_time']:.2f}s", 
                    f"  P95 Response Time: {test['p95_response_time']:.2f}s",
                    f"  Success Rate: {test['success_rate']:.1%}",
                    ""
                ])
        
        # Add efficiency metrics if available
        if 'inference_efficiency' in self.inference_metrics:
            report_lines.extend([
                "",
                "⚡ INFERENCE EFFICIENCY:",
                "-" * 40,
            ])
            
            for metric, score in self.inference_metrics['inference_efficiency'].items():
                report_lines.append(f"{metric}: {score:.3f}")
        
        # Add error details if any
        if self.inference_metrics.get('errors'):
            report_lines.extend([
                "",
                "❌ ERRORS:",
                "-" * 40,
            ])
            for error in self.inference_metrics['errors']:
                report_lines.append(f"  {error}")
        
        return "\n".join(report_lines)
    
    # Include necessary methods from original inference script
    def _load_recent_data(self, stock: str):
        """Load recent data for a stock from its main stock table"""
        table_path = (
            self.config.paths.delta_tables_dir / 
            f"{self.config.data.stock_tables_prefix}{stock}"
        )
        
        if not table_path.exists():
            raise FileNotFoundError(f"Stock table not found: {table_path}")
        
        df = self.spark.read.format("delta").load(str(table_path))
        
        df_recent = (df
                    .orderBy(col("Date").desc())
                    .limit(self.lookback_days + 10)
                    .orderBy(col("Date").asc())
                    )
        
        return df_recent
    
    def _validate_data_quality(self, df, stock: str) -> None:
        """Validate data quality before making predictions"""
        row_count = df.count()
        
        if row_count < self.config.inference.min_required_rows:
            raise ValueError(
                f"Insufficient data for {stock}: {row_count} rows, "
                f"need at least {self.config.inference.min_required_rows}"
            )
    
    def _prepare_features_for_stock(self, stock_df: DataFrame, stock_symbol: str) -> DataFrame:
        """
        Prepare features for a single stock using inference.py approach
        Creates features per stock separately like in the original inference script
        """
        self.logger.debug(f"Creating inference features for {stock_symbol}")
        
        try:
            # Create features using FeatureEngineer (same as training and inference.py)
            df_with_features, feature_names = self.feature_engineer.create_all_features(stock_df)
            
            self.logger.debug(f"Using {len(feature_names)} features for {stock_symbol}")
            
            # Get the latest row (most recent data for prediction)
            latest_df = df_with_features.orderBy(col("Date").desc()).limit(1)
            
            # Ensure all feature columns are double type (same as training)
            for feature_name in feature_names:
                latest_df = latest_df.withColumn(feature_name, col(feature_name).cast("double"))
            
            # Create VectorAssembler with the feature names
            assembler = VectorAssembler(inputCols=feature_names, outputCol="features_raw")
            feature_df = assembler.transform(latest_df)
            
            # Apply scaling using the loaded scaler
            if self.feature_scaler is None:
                raise ValueError("Feature scaler not loaded")
                
            scaled_df = self.feature_scaler.transform(feature_df)
            
            self.logger.debug(f"✅ Features prepared for {stock_symbol}")
            return scaled_df
            
        except Exception as e:
            self.logger.error(f"Error preparing features for {stock_symbol}: {e}")
            self.logger.error(f"Available columns: {stock_df.columns}")
            raise
    
    def _create_prediction_result(self, stock: str, prediction: float, recent_data: DataFrame) -> Dict:
        """
        Create prediction result dictionary with per-stock prediction details
        """
        # Calculate prediction metadata
        last_date = recent_data.select(spark_max("Date")).collect()[0][0]
        next_trading_date = self._get_next_trading_date(last_date)
        last_close = recent_data.orderBy(col("Date").desc()).select("Close").limit(1).collect()[0][0]
        
        # Validate prediction reasonableness
        if not self._validate_prediction_reasonableness(prediction, stock):
            self.logger.warning(f"Using fallback prediction for {stock}")
            prediction = 0.001  # 0.1% gain
        
        predicted_close = float(last_close * (1 + prediction))
        
        # Create per-stock prediction result
        prediction_result = {
            'stock': stock,
            'last_date': last_date,
            'next_trading_date': next_trading_date,
            'last_close': float(last_close),
            'predicted_close': predicted_close,
            'predicted_gain': float(prediction),
            'rows_used': recent_data.count(),
            'prediction_timestamp': datetime.now(),
            'confidence_score': self._calculate_prediction_confidence(prediction, stock),
            'model_type': 'unified',
            'model_name': self.model_name
        }
        
        return prediction_result
    def _prepare_features(self, stock_df: DataFrame, stock_symbol: str) -> DataFrame:
        """
        Prepare features for inference (backward compatibility method)
        Delegates to _prepare_features_for_stock for consistency
        """
        return self._prepare_features_for_stock(stock_df, stock_symbol)
    
    def _make_prediction(self, feature_df, stock: str) -> float:
        """Make price prediction using the loaded model"""
        try:
            prediction_df = self.model.transform(feature_df)
            prediction_value = prediction_df.select("prediction").collect()[0][0]
            return prediction_value
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {stock}: {str(e)}")
            raise
    
    def _save_prediction(self, prediction_result: Dict) -> None:
        """Save prediction to Delta table"""
        try:
            from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
            
            predictions_path = str(self.config.paths.delta_tables_dir / "predictions")
            
            try:
                # Check existing schema
                existing_df = self.spark.read.format("delta").load(predictions_path)
                existing_fields = [field.name for field in existing_df.schema.fields]
                use_legacy_schema = "current_date" in existing_fields
            except Exception:
                use_legacy_schema = False
            
            # Prepare data based on schema
            if use_legacy_schema:
                prediction_data = [{
                    "stock_symbol": prediction_result['stock'],
                    "current_date": prediction_result['prediction_timestamp'],
                    "current_price": prediction_result['last_close'],
                    "predicted_return": prediction_result['predicted_gain'],
                    "predicted_price": prediction_result['predicted_close'],
                    "timestamp": datetime.now()
                }]
            else:
                prediction_data = [{
                    "stock_symbol": prediction_result['stock'],
                    "prediction_date": prediction_result['prediction_timestamp'],
                    "predicted_for_date": prediction_result['next_trading_date'].strftime('%Y-%m-%d'),
                    "predicted_gain": prediction_result['predicted_gain'],
                    "predicted_close": prediction_result['predicted_close'],
                    "last_close": prediction_result['last_close'],
                    "model_used": self.model_name,
                    "model_version": str(self.model_version),
                    "rows_used": str(prediction_result['rows_used']),
                    "created_at": datetime.now(),
                    "confidence_score": prediction_result.get('confidence_score', 0.0)
                }]
            
            pred_df = self.spark.createDataFrame(prediction_data)
            
            if use_legacy_schema:
                pred_df.write.format("delta").mode("append").save(predictions_path)
            else:
                pred_df.write.format("delta").option("mergeSchema", "true").mode("append").save(predictions_path)
            
            self.logger.debug(f"✅ Prediction saved for {prediction_result['stock']}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save prediction for {prediction_result['stock']}: {e}")
    
    def _get_next_trading_date(self, last_date) -> datetime:
        """Calculate next trading date (skip weekends)"""
        if isinstance(last_date, str):
            last_date = datetime.strptime(last_date, "%Y-%m-%d")
        elif hasattr(last_date, 'date'):
            last_date = last_date.date()
            last_date = datetime.combine(last_date, datetime.min.time())
        
        next_date = last_date + timedelta(days=1)
        
        while next_date.weekday() >= 5:  # Skip weekends
            next_date += timedelta(days=1)
        
        return next_date
    
    def _log_inference_summary(self):
        """Log comprehensive inference summary"""
        print("\n" + "="*80)
        print("🔮 MONITORED INFERENCE COMPLETED")
        print("="*80)
        print(f"� Model Architecture: {self.inference_metrics['model_type']}")
        print(f"🏷️  Model Used: {self.inference_metrics['model_name_used']}")
        if self.is_unified_model:
            print(f"🔗 Unified Model Stocks: {', '.join(self.unified_stocks)}")
        print(f"�📊 Stocks Processed: {len(self.inference_metrics['stocks_processed'])}")
        print(f"⏱️  Total Duration: {self.inference_metrics['total_inference_time']:.2f} seconds")
        print(f"✅ Successful Predictions: {len(self.inference_metrics['predictions'])}")
        
        if self.inference_metrics['errors']:
            print(f"❌ Failed Predictions: {len(self.inference_metrics['errors'])}")
        
        # Print scalability metrics if available
        if 'scalability_metrics' in self.inference_metrics and self.inference_metrics['scalability_metrics']:
            metrics = self.inference_metrics['scalability_metrics']
            print(f"📈 Throughput: {metrics.throughput_records_per_second:.1f} predictions/sec")
            print(f"💾 Peak Memory: {metrics.peak_memory_usage_mb:.1f} MB")
            print(f"🎯 Scalability Score: {metrics.linear_scalability_score:.3f}")
        
        # Print efficiency metrics if available
        if 'inference_efficiency' in self.inference_metrics:
            print("\n⚡ EFFICIENCY SCORES:")
            for metric, score in self.inference_metrics['inference_efficiency'].items():
                print(f"   {metric}: {score:.3f}")
        
        # Print load test summary if available
        if 'load_test_results' in self.inference_metrics:
            print("\n🏃 LOAD TEST SUMMARY:")
            for test in self.inference_metrics['load_test_results']['concurrent_tests']:
                print(f"   {test['concurrency_level']} concurrent: {test['throughput']:.1f} req/sec")
        
        print("\n📈 PREDICTIONS:")
        print("-" * 80)
        for stock, pred_info in self.inference_metrics['predictions'].items():
            confidence = pred_info.get('confidence_score', 0)
            model_type = pred_info.get('model_type', 'unknown')
            print(f"{stock:6} | Last: ${pred_info['last_close']:.2f} | "
                  f"Predicted: ${pred_info['predicted_close']:.2f} | "
                  f"Gain: {pred_info['predicted_gain']:.4f} ({pred_info['predicted_gain']*100:.2f}%) | "
                  f"Confidence: {confidence:.2f} | Type: {model_type}")
        print("-" * 80)
        
        # Show model configuration
        print(f"\n🔧 MODEL CONFIGURATION:")
        print(f"   Model Name: {self.model_name}")
        print(f"   Model Version: {self.model_version}")
        print(f"   Model Stage: {self.model_stage}")
        print(f"   Lookback Days: {self.lookback_days}")
        
        if self.inference_metrics['errors']:
            print("\n⚠️ ERRORS:")
            for error in self.inference_metrics['errors']:
                print(f"   {error}")
        
        # Show prediction persistence status
        if self.config.inference.save_predictions:
            print(f"\n💾 Predictions saved to: delta_tables/predictions")
        else:
            print(f"\n💡 Predictions are for display only and not saved to tables.")
            print(f"   Set 'inference.save_predictions: true' in config to persist predictions.")


def create_spark_session(config: Config) -> SparkSession:
    """Create Spark session for inference with monitoring"""
    
    builder = SparkSession.builder \
        .appName(f"{config.spark.app_name}_Inference_With_Monitoring") \
        .master(config.spark.master) \
        .config("spark.driver.memory", config.spark.driver_memory) \
        .config("spark.executor.memory", config.spark.executor_memory) \
        .config("spark.executor.cores", config.spark.executor_cores) \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    
    # Add additional configurations
    for key, value in config.spark.configs.items():
        builder = builder.config(key, value)
    
    return configure_spark_with_delta_pip(builder).getOrCreate()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run inference with scalability monitoring')
    
    parser.add_argument(
        '--stocks', 
        type=str, 
        help='Comma-separated list of stock symbols for inference'
    )
    parser.add_argument(
        '--no-monitoring', 
        action='store_true', 
        help='Disable scalability monitoring'
    )
    parser.add_argument(
        '--load-test', 
        action='store_true', 
        help='Run concurrent load testing'
    )
    parser.add_argument(
        '--monitoring-config', 
        type=str, 
        help='Path to custom monitoring configuration JSON file'
    )
    
    return parser.parse_args()


def main():
    """Main inference pipeline with monitoring and unified model support"""
    
    print("🔮 Starting Monitored Inference Pipeline with Unified Model Support")
    print("=" * 80)
    
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Load environment and configuration
        load_environment()
        config = Config()
        
        # Override monitoring setting from command line
        if args.no_monitoring:
            config.monitoring.enabled = False
        
        # Create Spark session
        spark = create_spark_session(config)
        
        print("🔧 Spark session created successfully")
        
        # Create monitored inference engine with unified model support
        enable_monitoring = config.monitoring.enabled and not args.no_monitoring
        inference_engine = MonitoredInferenceEngine(config, spark, enable_monitoring)
        
        # Show model information
        model_type = "Unified" if inference_engine.is_unified_model else "Per-stock"
        print(f"📦 Model Architecture: {model_type}")
        print(f"🏷️  Model Name: {config.inference.model_name}")
        if inference_engine.is_unified_model:
            print(f"🔗 Unified Model Supports: {', '.join(inference_engine.unified_stocks)}")
        
        # Determine stock selection
        if args.stocks:
            stocks = [s.strip().upper() for s in args.stocks.split(',')]
            print(f"🎯 Running inference for specific stocks: {stocks}")
            
            # Validate stock symbols
            invalid_stocks = [s for s in stocks if s not in config.data.available_stocks]
            if invalid_stocks:
                print(f"❌ Invalid stock symbols: {invalid_stocks}")
                print(f"Available stocks: {config.data.available_stocks}")
                return
            
            # Check unified model compatibility
            if inference_engine.is_unified_model:
                unsupported_stocks = [s for s in stocks if s not in inference_engine.unified_stocks]
                if unsupported_stocks:
                    print(f"⚠️  Warning: Stocks not supported by unified model: {unsupported_stocks}")
                    print(f"Unified model supports: {inference_engine.unified_stocks}")
                    # Filter to only supported stocks
                    stocks = [s for s in stocks if s in inference_engine.unified_stocks]
                    if not stocks:
                        print("❌ No supported stocks found for unified model")
                        return
                    print(f"🔄 Proceeding with supported stocks: {stocks}")
        else:
            stocks = config.data.inference_stocks
            print(f"🎯 Running inference for configured stocks: {stocks}")
            
            # Filter stocks for unified model if needed
            if inference_engine.is_unified_model:
                original_stocks = stocks[:]
                stocks = [s for s in stocks if s in inference_engine.unified_stocks]
                if len(stocks) != len(original_stocks):
                    removed_stocks = [s for s in original_stocks if s not in stocks]
                    print(f"🔄 Unified model filtering: removed {removed_stocks}, using {stocks}")
        
        # Run monitored inference
        results = inference_engine.run_monitored_inference(stocks, args.load_test)
        
        print("\n🎉 Monitored inference completed successfully!")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Monitored inference failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # Stop Spark session
        if 'spark' in locals():
            spark.stop()
            print("🔧 Spark session stopped")


if __name__ == "__main__":
    main()