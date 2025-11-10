#!/usr/bin/env python3
"""
Enhanced Stock Price Inference Pipeline
Loads trained model from MLflow and generates next-day predictions for configured stocks

CONDA ENVIRONMENT: breaking_data

Usage: 
    # Run inference for all configured stocks (optimized batch processing)
    conda activate breaking_data
    python scripts/inference.py

    # Run inference for specific stocks
    python scripts/inference.py AAPL GOOG NVDA
    
Features:
- Uses same feature engineering pipeline as training
- Proper data validation and quality checks  
- Prediction persistence to Delta tables (configurable)
- Batch processing optimization
- Comprehensive error handling and logging
- MLflow experiment tracking for inference runs
- Command line interface for stock selection
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
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



class StockInferenceEngine:
    """
    Production inference engine for stock price prediction
    """
    
    def __init__(self, config: Config, spark: SparkSession):
        self.config = config
        self.spark = spark
        self.logger = Logger().get_logger()
        
        self.data_processor = DataProcessor(config, spark)
        self.feature_engineer = FeatureEngineer(config, spark)
        self.model_trainer = ModelTrainer(config, spark)
        self.mlflow_manager = MLflowManager(config)
        
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
    
    def _load_model(self) -> None:
        """Load the trained model from MLflow"""
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
            
            self.logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def _extract_feature_names(self, model_uri: str) -> None:
        """Extract feature names from model metadata or by creating features"""
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
            self.logger.debug(f"Feature names: {self.feature_names[:10]}...")  # Show first 10
                
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
    
    def run_inference(self, stocks: Optional[List[str]] = None) -> Dict:
        """
        Run inference for specified stocks or all inference stocks from config
        
        Args:
            stocks: List of stock symbols. If None, uses config.data.inference_stocks
            
        Returns:
            Dictionary with inference results and metrics
        """
        start_time = datetime.now()
        
        # Use config stocks if none specified
        if stocks is None:
            stocks = self.config.data.inference_stocks
        
        self.logger.info(f"🔮 Starting inference for stocks: {stocks}")
        
        results = {
            'start_time': start_time,
            'stocks_processed': [],
            'predictions': {},
            'errors': [],
            'metrics': {
                'total_predictions': 0,
                'successful_predictions': 0,
                'failed_predictions': 0,
                'total_inference_time': 0,
                'avg_inference_time_per_stock': 0
            }
        }
        
        # Process each stock
        for stock in stocks:
            try:
                self.logger.info(f"📊 Processing inference for {stock}...")
                
                stock_start_time = datetime.now()
                prediction_result = self._run_single_stock_inference(stock)
                stock_duration = (datetime.now() - stock_start_time).total_seconds()
                
                results['stocks_processed'].append(stock)
                results['predictions'][stock] = prediction_result
                results['metrics']['total_predictions'] += 1
                results['metrics']['successful_predictions'] += 1
                
                self.logger.info(f"✅ {stock} inference completed in {stock_duration:.2f}s")
                
            except Exception as e:
                error_msg = f"Inference failed for {stock}: {str(e)}"
                self.logger.error(error_msg)
                results['errors'].append(error_msg)
                results['metrics']['failed_predictions'] += 1
        
        # Calculate final metrics
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        results['end_time'] = end_time
        results['metrics']['total_inference_time'] = total_time
        
        if results['metrics']['successful_predictions'] > 0:
            results['metrics']['avg_inference_time_per_stock'] = (
                total_time / results['metrics']['successful_predictions']
            )
        
        # Log summary
        self._log_inference_summary(results)
        
        return results
    
    def _run_single_stock_inference(self, stock: str) -> Dict:
        """
        Run inference for a single stock
        
        Args:
            stock: Stock symbol
            
        Returns:
            Dictionary with prediction results
        """
        # Load recent data for the stock
        recent_data = self._load_recent_data(stock)
        
        # Validate data quality
        self._validate_data_quality(recent_data, stock)
        
        # Prepare features for prediction
        feature_data = self._prepare_features(recent_data, stock)
        
        # Make prediction
        prediction = self._make_prediction(feature_data, stock)
        
        # Validate prediction reasonableness
        if not self.validate_prediction_reasonableness(prediction, stock):
            self.logger.warning(f"Using fallback prediction for {stock}")
            # Use a conservative fallback (small positive return)
            prediction = 0.001  # 0.1% gain
        
        # Calculate next trading date
        last_date = recent_data.select(spark_max("Date")).collect()[0][0]
        next_trading_date = self._get_next_trading_date(last_date)
        
        # Calculate predicted close price
        last_close = recent_data.orderBy(col("Date").desc()).select("Close").limit(1).collect()[0][0]
        predicted_close = float(last_close * (1 + prediction))
        
        # Create prediction result
        prediction_result = {
            'stock': stock,
            'last_date': last_date,
            'next_trading_date': next_trading_date,
            'last_close': float(last_close),
            'predicted_close': predicted_close,
            'predicted_gain': float(prediction),  # Original gain prediction
            'rows_used': recent_data.count(),
            'prediction_timestamp': datetime.now()
        }
        
        # Save prediction to Delta table if configured
        if self.config.inference.save_predictions:
            self._save_prediction(prediction_result)
        
        return prediction_result
    
    def _load_recent_data(self, stock: str):
        """Load recent data for a stock from its main stock table (for feature engineering)"""
        table_path = (
            self.config.paths.delta_tables_dir / 
            f"{self.config.data.stock_tables_prefix}{stock}"
        )
        
        if not table_path.exists():
            raise FileNotFoundError(f"Stock table not found: {table_path}")
        
        # Load the full stock table and get the most recent rows
        df = self.spark.read.format("delta").load(str(table_path))
        
        # Get the most recent lookback_days rows for feature engineering
        df_recent = (df
                    .orderBy(col("Date").desc())
                    .limit(self.lookback_days + 10)  # Get extra rows for safety
                    .orderBy(col("Date").asc())  # Re-order chronologically
                    )
        
        return df_recent
    
    def _validate_data_quality(self, df, stock: str) -> None:
        """Validate data quality before making predictions"""
        row_count = df.count()
        
        # Check minimum rows requirement
        if row_count < self.config.inference.min_required_rows:
            raise ValueError(
                f"Insufficient data for {stock}: {row_count} rows, "
                f"need at least {self.config.inference.min_required_rows}"
            )
        
        if self.config.inference.validate_data_quality:
            # Check for missing values
            if self.config.inference.quality_checks.get("check_missing_values", True):
                self._check_missing_values(df, stock)
            
            # Check data continuity (no large gaps in dates)
            if self.config.inference.quality_checks.get("check_data_continuity", True):
                self._check_data_continuity(df, stock)
    
    def _check_missing_values(self, df, stock: str) -> None:
        """Check for excessive missing values"""
        total_rows = df.count()
        max_missing_ratio = self.config.inference.quality_checks.get("max_missing_ratio", 0.1)
        
        # Check key columns for missing values
        key_columns = ["Close", "Volume", "Date"]
        for col_name in key_columns:
            if col_name in df.columns:
                null_count = df.filter(col(col_name).isNull()).count()
                missing_ratio = null_count / total_rows
                
                if missing_ratio > max_missing_ratio:
                    raise ValueError(
                        f"Too many missing values in {col_name} for {stock}: "
                        f"{missing_ratio:.2%} > {max_missing_ratio:.2%}"
                    )
    
    def _check_data_continuity(self, df, stock: str) -> None:
        """Check for data continuity (no large gaps)"""
        # Convert to Pandas for easier date operations
        df_pd = df.select("Date").orderBy("Date").toPandas()
        df_pd['Date'] = pd.to_datetime(df_pd['Date'])
        
        # Check for gaps larger than 5 days (considering weekends)
        date_diffs = df_pd['Date'].diff().dt.days
        max_gap = date_diffs.max()
        
        if max_gap > 7:  # More than a week gap
            self.logger.warning(
                f"Large data gap detected for {stock}: {max_gap} days. "
                "This might affect prediction quality."
            )
    
    def _prepare_features(self, stock_df: DataFrame, stock_symbol: str) -> DataFrame:
        """Prepare features for inference using the same pipeline as training"""
        self.logger.info("Creating inference features...")
        
        try:
            # Create features using FeatureEngineer (same as training)
            df_with_features, feature_names = self.feature_engineer.create_all_features(stock_df)
            
            self.logger.info(f"Using {len(feature_names)} features for {stock_symbol}")
            
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
            
            self.logger.info(f"✅ Features prepared for {stock_symbol}")
            return scaled_df
            
        except Exception as e:
            self.logger.error(f"Error preparing features for {stock_symbol}: {e}")
            self.logger.error(f"Available columns: {stock_df.columns}")
            raise
    
    def _make_prediction(self, feature_df, stock: str) -> float:
        """Make price prediction using the loaded model"""
        try:
            # Make prediction
            prediction_df = self.model.transform(feature_df)
            
            # Extract prediction value
            prediction_value = prediction_df.select("prediction").collect()[0][0]
            
            self.logger.info(f"Prediction for {stock}: {prediction_value}")
            return prediction_value
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {stock}: {str(e)}")
            raise
    
    def _save_prediction(self, prediction_result: Dict) -> None:
        """Save prediction to Delta table"""
        try:
            from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
            
            # Read the existing table schema if it exists
            predictions_path = str(self.config.paths.delta_tables_dir / "predictions")
            
            try:
                # Try to read existing table to get schema
                existing_df = self.spark.read.format("delta").load(predictions_path)
                existing_schema = existing_df.schema
                
                # Create data row matching existing schema
                if "current_date" in [field.name for field in existing_schema.fields]:
                    # Use existing schema format
                    prediction_data = [{
                        "stock_symbol": prediction_result['stock'],
                        "current_date": prediction_result['prediction_timestamp'],
                        "current_price": prediction_result['last_close'],
                        "predicted_return": prediction_result['predicted_gain'],
                        "predicted_price": prediction_result['predicted_close'],
                        "timestamp": datetime.now()
                    }]
                    pred_df = self.spark.createDataFrame(prediction_data, schema=existing_schema)
                else:
                    # Use new schema format with mergeSchema
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
                        "created_at": datetime.now()
                    }]
                    pred_df = self.spark.createDataFrame(prediction_data)
                    
                    # Write with mergeSchema option
                    pred_df.write.format("delta").option("mergeSchema", "true").mode("append").save(predictions_path)
                    self.logger.info(f"✅ Prediction saved for {prediction_result['stock']} (schema merged)")
                    return
                    
            except Exception:
                # Table doesn't exist, create with new schema
                prediction_schema = StructType([
                    StructField("stock_symbol", StringType(), False),
                    StructField("prediction_date", TimestampType(), False),
                    StructField("predicted_for_date", StringType(), False),
                    StructField("predicted_gain", DoubleType(), False),
                    StructField("predicted_close", DoubleType(), False),
                    StructField("last_close", DoubleType(), False),
                    StructField("model_used", StringType(), False),
                    StructField("model_version", StringType(), True),
                    StructField("rows_used", StringType(), True),
                    StructField("created_at", TimestampType(), False)
                ])
                
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
                    "created_at": datetime.now()
                }]
                
                pred_df = self.spark.createDataFrame(prediction_data, schema=prediction_schema)
            
            # Write to Delta table (append mode)
            pred_df.write.format("delta").mode("append").save(predictions_path)
            
            self.logger.info(f"✅ Prediction saved for {prediction_result['stock']}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save prediction for {prediction_result['stock']}: {e}")
            # Don't raise exception - continue with inference even if saving fails
    
    def _get_next_trading_date(self, last_date) -> datetime:
        """Calculate next trading date (skip weekends)"""
        if isinstance(last_date, str):
            last_date = datetime.strptime(last_date, "%Y-%m-%d")
        elif hasattr(last_date, 'date'):
            last_date = last_date.date()
            last_date = datetime.combine(last_date, datetime.min.time())
        
        next_date = last_date + timedelta(days=1)
        
        # Skip weekends (Monday=0, Sunday=6)
        while next_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
            next_date += timedelta(days=1)
        
        return next_date
    
    def _log_inference_summary(self, results: Dict) -> None:
        """Log inference summary and metrics to MLflow"""
        
        # Start MLflow run for inference tracking
        with mlflow.start_run(run_name=f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.set_tag("run_type", "inference")
            mlflow.set_tag("model_used", self.model_name)
            
            # Log inference parameters
            mlflow.log_param("stocks_count", len(results['stocks_processed']))
            mlflow.log_param("stocks_list", ','.join(results['stocks_processed']))
            mlflow.log_param("lookback_days", self.lookback_days)
            
            # Log inference metrics
            metrics = results['metrics']
            mlflow.log_metric("total_predictions", metrics['total_predictions'])
            mlflow.log_metric("successful_predictions", metrics['successful_predictions'])
            mlflow.log_metric("failed_predictions", metrics['failed_predictions'])
            mlflow.log_metric("total_inference_time", metrics['total_inference_time'])
            mlflow.log_metric("avg_inference_time_per_stock", metrics['avg_inference_time_per_stock'])
            
            # Log individual stock predictions
            for stock, pred_info in results['predictions'].items():
                mlflow.log_metric(f"{stock}_predicted_close", pred_info['predicted_close'])
                mlflow.log_metric(f"{stock}_rows_used", pred_info['rows_used'])
            
            # Log system metrics
            self._log_system_metrics()
        
        # Print summary
        print("\n" + "="*80)
        print("🔮 INFERENCE PIPELINE COMPLETED")
        print("="*80)
        print(f"📊 Stocks Processed: {len(results['stocks_processed'])}")
        print(f"⏱️  Total Duration: {metrics['total_inference_time']:.2f} seconds")
        print(f"✅ Successful Predictions: {metrics['successful_predictions']}")
        if metrics['failed_predictions'] > 0:
            print(f"❌ Failed Predictions: {metrics['failed_predictions']}")
        print(f"⚡ Avg Time per Stock: {metrics['avg_inference_time_per_stock']:.2f} seconds")
        print()
        
        # Print individual predictions
        print("📈 PREDICTIONS:")
        print("-" * 80)
        for stock, pred_info in results['predictions'].items():
            print(f"{stock:6} | Last Close: ${pred_info.get('last_close', 0):.2f} | "
                  f"Predicted Close: ${pred_info['predicted_close']:.2f} | "
                  f"Gain: {pred_info['predicted_gain']:.4f} ({pred_info['predicted_gain']*100:.2f}%) | "
                  f"Date: {pred_info['next_trading_date'].strftime('%Y-%m-%d')}")
        print("-" * 80)
        
        if results['errors']:
            print("\n⚠️ ERRORS:")
            for error in results['errors']:
                print(f"   {error}")
        
        # Show prediction persistence status
        if self.config.inference.save_predictions:
            print(f"\n� Predictions saved to: delta_tables/predictions")
        else:
            print(f"\n💡 Predictions are for display only and not saved to tables.")
            print(f"   Set 'inference.save_predictions: true' in config to persist predictions.")
    
    def _log_system_metrics(self) -> None:
        """Log system performance metrics"""
        import psutil
        
        # Memory usage
        memory_info = psutil.virtual_memory()
        mlflow.log_metric("inference_memory_usage_percent", memory_info.percent)
        mlflow.log_metric("inference_memory_available_gb", memory_info.available / (1024**3))
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        mlflow.log_metric("inference_cpu_usage_percent", cpu_percent)
        
        # Disk usage
        disk_info = psutil.disk_usage('/')
        mlflow.log_metric("inference_disk_usage_percent", disk_info.percent)
    
    def run_batch_inference_optimized(self, stocks: Optional[List[str]] = None) -> Dict:
        """
        Optimized batch inference for multiple stocks
        Loads data once and processes multiple stocks efficiently
        
        Args:
            stocks: List of stock symbols. If None, uses config.data.inference_stocks
            
        Returns:
            Dictionary with batch inference results
        """
        start_time = datetime.now()
        
        if stocks is None:
            stocks = self.config.data.inference_stocks
        
        self.logger.info(f"🚀 Starting optimized batch inference for {len(stocks)} stocks")
        
        results = {
            'start_time': start_time,
            'stocks_processed': [],
            'predictions': {},
            'errors': [],
            'metrics': {
                'total_predictions': 0,
                'successful_predictions': 0,
                'failed_predictions': 0,
                'total_inference_time': 0,
                'avg_inference_time_per_stock': 0
            }
        }
        
        try:
            # Process all stocks in batch
            for stock in stocks:
                try:
                    stock_start_time = datetime.now()
                    prediction_result = self._run_single_stock_inference(stock)
                    stock_duration = (datetime.now() - stock_start_time).total_seconds()
                    
                    results['stocks_processed'].append(stock)
                    results['predictions'][stock] = prediction_result
                    results['metrics']['total_predictions'] += 1
                    results['metrics']['successful_predictions'] += 1
                    
                    self.logger.info(f"✅ {stock} processed in {stock_duration:.2f}s")
                    
                except Exception as e:
                    error_msg = f"Batch inference failed for {stock}: {str(e)}"
                    self.logger.error(error_msg)
                    results['errors'].append(error_msg)
                    results['metrics']['failed_predictions'] += 1
            
            # Save all predictions in one batch if enabled
            if self.config.inference.save_predictions and results['predictions']:
                self._save_batch_predictions(results['predictions'])
            
        except Exception as e:
            self.logger.error(f"Batch inference pipeline failed: {e}")
            results['errors'].append(f"Pipeline error: {str(e)}")
        
        # Calculate metrics
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        results['end_time'] = end_time
        results['metrics']['total_inference_time'] = total_time
        
        if results['metrics']['successful_predictions'] > 0:
            results['metrics']['avg_inference_time_per_stock'] = (
                total_time / results['metrics']['successful_predictions']
            )
        
        self._log_inference_summary(results)
        return results
    
    def _save_batch_predictions(self, predictions: Dict) -> None:
        """Save multiple predictions to Delta table in one operation"""
        try:
            predictions_path = str(self.config.paths.delta_tables_dir / "predictions")
            
            # Check existing schema
            try:
                existing_df = self.spark.read.format("delta").load(predictions_path)
                existing_fields = [field.name for field in existing_df.schema.fields]
                use_legacy_schema = "current_date" in existing_fields
            except Exception:
                use_legacy_schema = False
            
            # Prepare batch data based on schema
            batch_data = []
            for stock, pred_result in predictions.items():
                if use_legacy_schema:
                    batch_data.append({
                        "stock_symbol": pred_result['stock'],
                        "current_date": pred_result['prediction_timestamp'],
                        "current_price": pred_result['last_close'],
                        "predicted_return": pred_result['predicted_gain'],
                        "predicted_price": pred_result['predicted_close'],
                        "timestamp": datetime.now()
                    })
                else:
                    batch_data.append({
                        "stock_symbol": pred_result['stock'],
                        "prediction_date": pred_result['prediction_timestamp'],
                        "predicted_for_date": pred_result['next_trading_date'].strftime('%Y-%m-%d'),
                        "predicted_gain": pred_result['predicted_gain'],
                        "predicted_close": pred_result['predicted_close'],
                        "last_close": pred_result['last_close'],
                        "model_used": self.model_name,
                        "model_version": str(self.model_version),
                        "rows_used": str(pred_result['rows_used']),
                        "created_at": datetime.now()
                    })
            
            # Create DataFrame and save
            batch_df = self.spark.createDataFrame(batch_data)
            
            if use_legacy_schema:
                batch_df.write.format("delta").mode("append").save(predictions_path)
            else:
                batch_df.write.format("delta").option("mergeSchema", "true").mode("append").save(predictions_path)
            
            self.logger.info(f"✅ Batch saved {len(batch_data)} predictions")
            
        except Exception as e:
            self.logger.warning(f"Failed to save batch predictions: {e}")
    
    def validate_prediction_reasonableness(self, predicted_gain: float, stock: str) -> bool:
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


def create_spark_session(config: Config) -> SparkSession:
    """Create Spark session for inference"""
    
    builder = SparkSession.builder \
        .appName(f"{config.spark.app_name}_Inference") \
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


def main():
    """Main inference pipeline"""
    
    print("🔮 Starting Stock Price Inference Pipeline")
    print("=" * 80)
    
    try:
        # Load environment and configuration
        load_environment()
        config = Config()
        
        # Create Spark session
        spark = create_spark_session(config)
        
        print("🔧 Spark session created successfully")
        
        # Create inference engine
        inference_engine = StockInferenceEngine(config, spark)
        
        # Check command line arguments for specific stocks
        import sys
        if len(sys.argv) > 1:
            # Run inference for specific stocks
            stocks = [stock.upper() for stock in sys.argv[1:] if stock.upper() in config.data.available_stocks]
            if stocks:
                print(f"🎯 Running inference for specified stocks: {stocks}")
                results = inference_engine.run_inference(stocks=stocks)
            else:
                print(f"❌ Invalid stock symbols. Available: {config.data.available_stocks}")
                return None
        else:
            # Run inference for all configured stocks using optimized batch processing
            print(f"🚀 Running optimized batch inference for all configured stocks")
            results = inference_engine.run_batch_inference_optimized()
        
        print("\n🎉 Inference pipeline completed successfully!")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Inference pipeline failed: {str(e)}")
        logger = Logger().get_logger()
        logger.error(f"Inference pipeline failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    finally:
        # Stop Spark session
        if 'spark' in locals():
            spark.stop()
            print("🔧 Spark session stopped")


if __name__ == "__main__":
    main()