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


class MonitoredInferenceEngine:
    """
    Production inference engine with comprehensive scalability monitoring
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
        
        # Inference settings from config
        self.lookback_days = config.inference.lookback_days
        self.model_name = config.inference.model_name
        self.model_stage = config.inference.model_stage
        self.model_version = config.inference.model_version
        
        # Load model and scaler on initialization
        self.model = None
        self.feature_scaler = None
        self.feature_names = None
        self._load_model()
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
            'scalability_metrics': None
        }
    
    def _load_model(self) -> None:
        """Load the trained model from MLflow with monitoring"""
        model_load_start = time.time()
        
        try:
            self.logger.info(f"Loading model {self.model_name} from MLflow...")
            
            # Set MLflow tracking URI
            mlflow.set_tracking_uri(str(self.config.paths.mlflow_tracking_dir))
            
            # Load model based on version specification
            if self.model_version == "latest":
                client = MlflowClient()
                model_version = client.get_latest_versions(
                    self.model_name, 
                    stages=[self.model_stage] if self.model_stage != "None" else None
                )[0]
                model_uri = f"models:/{self.model_name}/{model_version.version}"
            else:
                model_uri = f"models:/{self.model_name}/{self.model_version}"
            
            self.logger.info(f"Loading model from URI: {model_uri}")
            self.model = mlflow.spark.load_model(model_uri)
            
            # Extract feature names from model metadata
            self._extract_feature_names(model_uri)
            
            # Monitor model loading performance
            if self.monitor:
                self.monitor.monitor_performance("model_loading", model_load_start, 1)
            
            self.logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
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
            
            # Run inference with monitoring
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
                
                # Generate monitoring report
                self._generate_inference_report(scalability_metrics)
            
            self.inference_metrics['end_time'] = datetime.now()
            self.inference_metrics['total_inference_time'] = (
                self.inference_metrics['end_time'] - self.inference_metrics['start_time']
            ).total_seconds()
        
        # Log summary
        self._log_inference_summary()
        
        return self.inference_metrics
    
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
        
        # Calculate prediction metadata
        last_date = recent_data.select(spark_max("Date")).collect()[0][0]
        next_trading_date = self._get_next_trading_date(last_date)
        last_close = recent_data.orderBy(col("Date").desc()).select("Close").limit(1).collect()[0][0]
        predicted_close = float(last_close * (1 + prediction))
        
        # Validate prediction reasonableness
        if not self._validate_prediction_reasonableness(prediction, stock):
            self.logger.warning(f"Using fallback prediction for {stock}")
            prediction = 0.001  # 0.1% gain
            predicted_close = float(last_close * (1 + prediction))
        
        # Create prediction result
        prediction_result = {
            'stock': stock,
            'last_date': last_date,
            'next_trading_date': next_trading_date,
            'last_close': float(last_close),
            'predicted_close': predicted_close,
            'predicted_gain': float(prediction),
            'rows_used': recent_data.count(),
            'prediction_timestamp': datetime.now(),
            'confidence_score': self._calculate_prediction_confidence(prediction, stock)
        }
        
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
        """Calculate inference efficiency metrics"""
        if not self.monitor or not self.monitor.metrics:
            return
            
        metrics = self.monitor.metrics
        
        # Calculate efficiency scores
        inference_efficiency = {
            'latency_efficiency': self._calculate_latency_efficiency(),
            'throughput_efficiency': self._calculate_throughput_efficiency(),
            'resource_efficiency': self._calculate_resource_efficiency(),
            'prediction_quality_efficiency': self._calculate_prediction_quality_efficiency()
        }
        
        self.inference_metrics['inference_efficiency'] = inference_efficiency
        
        # Log efficiency warnings
        for metric, value in inference_efficiency.items():
            if value < 0.6:  # Below 60% efficiency
                self.logger.warning(f"⚠️  Low {metric}: {value:.2f}")
    
    def _calculate_latency_efficiency(self) -> float:
        """Calculate latency efficiency score"""
        if not self.monitor.metrics.avg_processing_time_per_record:
            return 0.0
            
        # Target: < 1 second per prediction
        target_latency_ms = 1000
        actual_latency_ms = self.monitor.metrics.avg_processing_time_per_record
        
        return min(1.0, target_latency_ms / max(actual_latency_ms, 1))
    
    def _calculate_throughput_efficiency(self) -> float:
        """Calculate throughput efficiency score"""
        if not self.monitor.metrics.throughput_records_per_second:
            return 0.0
            
        # Target: > 10 predictions per second
        target_throughput = 10
        actual_throughput = self.monitor.metrics.throughput_records_per_second
        
        return min(1.0, actual_throughput / target_throughput)
    
    def _calculate_resource_efficiency(self) -> float:
        """Calculate resource efficiency score"""
        if not self.monitor.metrics.avg_memory_usage_mb or not self.monitor.metrics.avg_cpu_usage_percent:
            return 0.0
            
        # Optimal resource usage
        memory_efficiency = self._calculate_memory_utilization_score()
        cpu_efficiency = self._calculate_cpu_utilization_score()
        
        return (memory_efficiency + cpu_efficiency) / 2
    
    def _calculate_memory_utilization_score(self) -> float:
        """Calculate memory utilization efficiency"""
        usage_mb = self.monitor.metrics.avg_memory_usage_mb
        total_memory_mb = 32 * 1024  # Assume 32GB
        usage_percent = (usage_mb / total_memory_mb) * 100
        
        # Optimal: 50-80% utilization
        if usage_percent < 30:
            return usage_percent / 50  # Underutilization
        elif usage_percent > 90:
            return (100 - usage_percent) / 10  # Overutilization  
        else:
            return 1.0  # Optimal range
    
    def _calculate_cpu_utilization_score(self) -> float:
        """Calculate CPU utilization efficiency"""
        usage_percent = self.monitor.metrics.avg_cpu_usage_percent
        
        # Optimal: 40-70% utilization for inference
        if usage_percent < 20:
            return usage_percent / 40
        elif usage_percent > 85:
            return (100 - usage_percent) / 15
        else:
            return 1.0
    
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
            "📊 INFERENCE RESULTS:",
            "-" * 40,
            f"Stocks Processed: {len(self.inference_metrics['stocks_processed'])}",
            f"Successful Predictions: {len(self.inference_metrics['predictions'])}",
            f"Failed Predictions: {len(self.inference_metrics['errors'])}",
        ]
        
        # Add individual predictions
        for stock, prediction in self.inference_metrics.get('predictions', {}).items():
            confidence = prediction.get('confidence_score', 0)
            report_lines.extend([
                f"",
                f"📈 {stock}:",
                f"  Last Close: ${prediction['last_close']:.2f}",
                f"  Predicted Close: ${prediction['predicted_close']:.2f}",
                f"  Predicted Gain: {prediction['predicted_gain']:.4f} ({prediction['predicted_gain']*100:.2f}%)",
                f"  Confidence: {confidence:.2f}",
                f"  Next Trading Date: {prediction['next_trading_date'].strftime('%Y-%m-%d') if hasattr(prediction['next_trading_date'], 'strftime') else prediction['next_trading_date']}",
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
    
    def _prepare_features(self, stock_df: DataFrame, stock_symbol: str) -> DataFrame:
        """Prepare features for inference using the same pipeline as training"""
        self.logger.debug(f"Creating inference features for {stock_symbol}")
        
        try:
            # Create features using FeatureEngineer (same as training)
            df_with_features, feature_names = self.feature_engineer.create_all_features(stock_df)
            
            # Get the latest row (most recent data for prediction)
            latest_df = df_with_features.orderBy(col("Date").desc()).limit(1)
            
            # Ensure all feature columns are double type
            for feature_name in feature_names:
                latest_df = latest_df.withColumn(feature_name, col(feature_name).cast("double"))
            
            # Create VectorAssembler
            assembler = VectorAssembler(inputCols=feature_names, outputCol="features_raw")
            feature_df = assembler.transform(latest_df)
            
            # Apply scaling
            if self.feature_scaler is None:
                raise ValueError("Feature scaler not loaded")
                
            scaled_df = self.feature_scaler.transform(feature_df)
            
            return scaled_df
            
        except Exception as e:
            self.logger.error(f"Error preparing features for {stock_symbol}: {e}")
            raise
    
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
        print(f"📊 Stocks Processed: {len(self.inference_metrics['stocks_processed'])}")
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
            print(f"{stock:6} | Last: ${pred_info['last_close']:.2f} | "
                  f"Predicted: ${pred_info['predicted_close']:.2f} | "
                  f"Gain: {pred_info['predicted_gain']:.4f} ({pred_info['predicted_gain']*100:.2f}%) | "
                  f"Confidence: {confidence:.2f}")
        print("-" * 80)


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
    """Main inference pipeline with monitoring"""
    
    print("🔮 Starting Monitored Inference Pipeline")
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
        
        # Create monitored inference engine
        enable_monitoring = config.monitoring.enabled and not args.no_monitoring
        inference_engine = MonitoredInferenceEngine(config, spark, enable_monitoring)
        
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
        else:
            stocks = config.data.inference_stocks
            print(f"🎯 Running inference for configured stocks: {stocks}")
        
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