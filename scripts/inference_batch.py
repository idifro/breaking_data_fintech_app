#!/usr/bin/env python3
"""
Inference Batch Script for Spark ML Pipeline
Loads Spark RF model from MLflow and makes predictions for 25 stocks

CONDA ENVIRONMENT: breaking_data
Usage: 
    conda activate breaking_data
    python scripts/inference_batch.py
"""

import sys
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add src to path
pipeline_root = Path(__file__).parent.parent
sys.path.insert(0, str(pipeline_root))

import warnings
warnings.filterwarnings("ignore")

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.types import *
from delta import configure_spark_with_delta_pip
import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient

# Local imports
from src.utils import Config, Logger, PathManager, load_environment
from src.data_processing import DataProcessor
from src.feature_engineering import FeatureEngineer
from scripts.train_model_with_modes import MultiModeConfig


class InferenceBatchProcessor:
    """
    Inference batch processor for 25 stocks using Spark RF model
    Following the same flow as train_model_with_modes.py
    """
    
    def __init__(self, multi_config: MultiModeConfig, spark: SparkSession):
        """Initialize inference batch processor"""
        self.multi_config = multi_config
        self.spark = spark
        self.logger = Logger().get_logger()
        self.path_manager = PathManager()
        
        # Get inference configuration
        self.inference_config = self._get_inference_config()
        
        # Initialize components with base config
        self.base_config = multi_config.base_config
        self.data_processor = DataProcessor(self.base_config, spark)
        self.feature_engineer = FeatureEngineer(self.base_config, spark)
        
        # MLflow client
        self.mlflow_client = MlflowClient()
        
        # Performance tracking
        self.start_time = None
        self.end_time = None
        self.performance_metrics = {}
        
        self.logger.info("🚀 Inference Batch Processor initialized")
        
    def _get_inference_config(self) -> Dict:
        """Get inference configuration from multi_config"""
        if 'inference_modes' not in self.multi_config._config:
            raise ValueError("Inference modes configuration not found in config_modes.yaml")
        
        if 'spark_rf_inference' not in self.multi_config._config['inference_modes']:
            raise ValueError("Spark RF inference configuration not found")
            
        return self.multi_config._config['inference_modes']['spark_rf_inference']
    
    def run_batch_inference(self) -> Dict[str, Any]:
        """Run complete batch inference pipeline"""
        
        self.logger.info("🔮 Starting batch inference for 25 stocks")
        self.start_time = time.time()
        
        try:
            # Step 1: Load model and scaler from MLflow
            self.logger.info("📥 Loading model and scaler from MLflow...")
            model, feature_scaler, model_info = self._load_model_and_scaler()
            
            # Step 2: Prepare inference data for 25 stocks
            self.logger.info("📊 Preparing inference data...")
            inference_df, feature_names = self._prepare_inference_data()
            
            # Step 3: Apply feature scaling
            self.logger.info("⚖️ Applying feature scaling...")
            scaled_df = self._apply_feature_scaling(inference_df, feature_scaler, feature_names)
            
            # Step 4: Make predictions
            self.logger.info("🎯 Making predictions...")
            predictions_df = self._make_predictions(model, scaled_df, feature_names, model_info)
            
            # Step 5: Calculate predicted close prices
            self.logger.info("💰 Calculating predicted close prices...")
            final_predictions_df = self._calculate_predicted_close_prices(predictions_df)
            
            # Step 6: Save predictions to Delta table
            self.logger.info("💾 Saving predictions to Delta table...")
            self._save_predictions_to_delta(final_predictions_df)
            
            # Step 7: Save to CSV (optional)
            if self.inference_config['output']['save_to_csv']:
                self.logger.info("📄 Saving predictions to CSV...")
                self._save_predictions_to_csv(final_predictions_df)
            
            # Calculate performance metrics
            self.end_time = time.time()
            self.performance_metrics = self._calculate_performance_metrics()
            
            # Log performance summary
            self._log_performance_summary()
            
            return {
                'status': 'success',
                'predictions_count': final_predictions_df.count(),
                'stocks_processed': len(self.inference_config['inference_stocks']),
                'performance_metrics': self.performance_metrics,
                'model_info': model_info
            }
            
        except Exception as e:
            self.logger.error(f"❌ Batch inference failed: {str(e)}")
            raise
    
    def _load_model_and_scaler(self) -> Tuple[object, object, Dict]:
        """Load Spark RF model and feature scaler from MLflow"""
        
        mlflow_config = self.inference_config['mlflow']
        model_name = mlflow_config['model_name']
        version = mlflow_config['version']
        
        try:
            # Load model from MLflow Model Registry
            if version == "latest":
                model_uri = f"models:/{model_name}/Latest"
            else:
                model_uri = f"models:/{model_name}/{version}"
            
            self.logger.info(f"📥 Loading model: {model_uri}")
            model = mlflow.spark.load_model(model_uri)
            
            # Get model version info
            if version == "latest":
                latest_version = self.mlflow_client.get_latest_versions(model_name, stages=["None", "Production", "Staging"])[0]
                model_version = latest_version.version
                run_id = latest_version.run_id
            else:
                model_version = version
                model_details = self.mlflow_client.get_model_version(model_name, version)
                run_id = model_details.run_id
            
            # Load feature scaler from model artifacts
            # The scaler is stored as a Spark transformer, not an MLflow model
            preprocessing_artifacts = self.mlflow_client.list_artifacts(run_id, "preprocessing")
            
            feature_scaler = None
            for artifact in preprocessing_artifacts:
                if "feature_scaler" in artifact.path:
                    # Download the scaler artifacts to a temporary location
                    import tempfile
                    temp_dir = tempfile.mkdtemp()
                    scaler_local_path = self.mlflow_client.download_artifacts(run_id, artifact.path, temp_dir)
                    
                    self.logger.info(f"📥 Loading feature scaler from: {scaler_local_path}")
                    
                    # Load as Spark transformer model
                    from pyspark.ml.feature import StandardScalerModel
                    feature_scaler = StandardScalerModel.load(scaler_local_path)
                    break
            
            if feature_scaler is None:
                raise ValueError("Feature scaler not found in model artifacts. Please ensure the model was trained with feature scaling.")
            
            model_info = {
                'model_name': model_name,
                'model_version': model_version,
                'run_id': run_id,
                'model_uri': model_uri
            }
            
            self.logger.info(f"✅ Model loaded successfully: {model_name} v{model_version}")
            return model, feature_scaler, model_info
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load model or scaler: {str(e)}")
            raise
    
    def _prepare_inference_data(self) -> Tuple[DataFrame, List[str]]:
        """Prepare inference data for 25 stocks - last 50 rows from each Delta table"""
        
        inference_stocks = self.inference_config['inference_stocks']
        lookback_days = self.inference_config['prediction_config']['lookback_days']
        
        self.logger.info(f"📊 Preparing data for {len(inference_stocks)} stocks, lookback: {lookback_days} days")
        
        # Load data from Delta tables for specified stocks only
        df = self.data_processor.load_data_from_delta()
        
        # Filter to only inference stocks
        df_filtered = df.filter(col("stock_symbol").isin(inference_stocks))
        
        # Get last lookback_days rows for each stock
        window_spec = Window.partitionBy("stock_symbol").orderBy(desc("Date"))
        latest_df = df_filtered.withColumn("row_num", row_number().over(window_spec)) \
                              .filter(col("row_num") <= lookback_days) \
                              .drop("row_num") \
                              .orderBy("stock_symbol", "Date")
        
        # Apply feature engineering (same as training)
        self.logger.info("🔧 Applying feature engineering...")
        df_with_features, feature_names = self.feature_engineer.create_all_features(latest_df)
        
        # Filter out rows with null features
        for feature_name in feature_names:
            df_with_features = df_with_features.filter(col(feature_name).isNotNull())
        
        # Get the latest row for each stock (for prediction)
        window_latest = Window.partitionBy("stock_symbol").orderBy(desc("Date"))
        final_df = df_with_features.withColumn("row_num", row_number().over(window_latest)) \
                                  .filter(col("row_num") == 1) \
                                  .drop("row_num")
        
        prediction_count = final_df.count()
        self.logger.info(f"✅ Prepared {prediction_count} records for prediction")
        
        return final_df, feature_names
    
    def _apply_feature_scaling(self, df: DataFrame, feature_scaler: object, feature_names: List[str]) -> DataFrame:
        """Apply feature scaling using the trained scaler"""
        
        from pyspark.ml.feature import VectorAssembler
        
        try:
            # Assemble features into vector
            assembler = VectorAssembler(
                inputCols=feature_names,
                outputCol="features_raw",
                handleInvalid="skip"
            )
            df_assembled = assembler.transform(df)
            
            # Apply scaling using the loaded scaler
            scaled_df = feature_scaler.transform(df_assembled)
            
            self.logger.info("✅ Feature scaling applied successfully")
            return scaled_df
            
        except Exception as e:
            self.logger.error(f"❌ Feature scaling failed: {str(e)}")
            raise
    
    def _make_predictions(self, model: object, scaled_df: DataFrame, feature_names: List[str], model_info: Dict) -> DataFrame:
        """Make predictions using the loaded Spark RF model"""
        
        try:
            # Make predictions
            predictions_df = model.transform(scaled_df)
            
            # Add metadata columns
            current_timestamp = datetime.now()
            
            predictions_with_meta = predictions_df.select(
                col("stock_symbol"),
                col("Date").alias("last_date"),
                col("Close").alias("last_close"),
                col("prediction").alias("predicted_gain"),
                lit(self.inference_config['prediction_config']['model_confidence']).alias("model_confidence"),
                lit(model_info['model_version']).alias("model_version"),
                lit(','.join(feature_names)).alias("features_used"),
                lit(current_timestamp).alias("created_at"),
                lit(1).alias("days_ahead")
            )
            
            self.logger.info("✅ Predictions generated successfully")
            return predictions_with_meta
            
        except Exception as e:
            self.logger.error(f"❌ Prediction failed: {str(e)}")
            raise
    
    def _calculate_predicted_close_prices(self, predictions_df: DataFrame) -> DataFrame:
        """Calculate predicted close prices using close gain: predicted_close = last_close * (1 + predicted_gain)"""
        
        try:
            # Calculate next business day
            from pyspark.sql.functions import udf
            from pyspark.sql.types import DateType, TimestampType
            
            def get_next_business_day(date_val):
                """Get next business day (skip weekends)"""
                if date_val is None:
                    return None
                #print("Date Received", date_val)
                next_day = date_val + timedelta(days=2)
                
                # Skip weekends
                while next_day.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                    next_day += timedelta(days=2)
                
                return next_day
            
            def get_next_business_timestamp(date_val):
                """Get next business day as timestamp"""
                if date_val is None:
                    return None
                
                next_day = get_next_business_day(date_val)
                # Set to market open time (9:30 AM EST)
                return datetime.combine(next_day, datetime.min.time().replace(hour=9, minute=30))
            
            next_business_day_udf = udf(get_next_business_day, DateType())
            next_business_timestamp_udf = udf(get_next_business_timestamp, TimestampType())
            
            # Calculate predicted close price and add prediction dates
            final_predictions = predictions_df.withColumn(
                "predicted_close", 
                col("last_close") * (lit(1) + col("predicted_gain"))
            ).withColumn(
                "prediction_date",
                next_business_day_udf(col("last_date"))
            ).withColumn(
                "prediction_timestamp",
                next_business_timestamp_udf(col("last_date"))
            ).select(
                col("prediction_date"),
                col("prediction_timestamp"), 
                col("predicted_close"),
                col("model_confidence"),
                col("model_version"),
                col("features_used"),
                col("created_at"),
                col("days_ahead"),
                col("stock_symbol")
            )
            
            self.logger.info("✅ Predicted close prices calculated successfully")
            return final_predictions
            
        except Exception as e:
            self.logger.error(f"❌ Close price calculation failed: {str(e)}")
            raise
    
    def _save_predictions_to_delta(self, predictions_df: DataFrame):
        """Save predictions to Delta table with specified schema"""
        
        try:
            # Debug: Check for duplicate columns
            columns = predictions_df.columns
            duplicate_columns = [col for col in columns if columns.count(col) > 1]
            if duplicate_columns:
                self.logger.error(f"❌ Found duplicate columns: {duplicate_columns}")
                self.logger.info(f"🔍 All columns: {columns}")
                raise ValueError(f"Duplicate columns found: {duplicate_columns}")
                
            delta_table_path = self.inference_config['output']['delta_table_path']
            
            # Ensure directory exists
            Path(delta_table_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Write to Delta table (append mode for incremental predictions)
            predictions_df.write \
                          .format("delta") \
                          .mode("append") \
                          .option("mergeSchema", "true") \
                          .save(delta_table_path)
            
            self.logger.info(f"✅ Predictions saved to Delta table: {delta_table_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save to Delta table: {str(e)}")
            raise
    
    def _save_predictions_to_csv(self, predictions_df: DataFrame):
        """Save predictions to CSV file"""
        
        try:
            csv_path = self.inference_config['output']['csv_path']
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Ensure directory exists
            Path(csv_path).mkdir(parents=True, exist_ok=True)
            
            # Convert to Pandas and save
            pandas_df = predictions_df.toPandas()
            csv_file = Path(csv_path) / f"batch_predictions_{timestamp}.csv"
            pandas_df.to_csv(csv_file, index=False)
            
            self.logger.info(f"✅ Predictions saved to CSV: {csv_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save to CSV: {str(e)}")
            raise
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics for monitoring"""
        
        if not self.start_time or not self.end_time:
            return {}
        
        import builtins
        
        total_time = self.end_time - self.start_time
        stocks_count = len(self.inference_config['inference_stocks'])
        
        metrics = {
            'total_inference_time_seconds': builtins.round(float(total_time), 2),
            'total_inference_time_minutes': builtins.round(float(total_time) / 60, 2),
            'stocks_processed': stocks_count,
            'avg_time_per_stock_seconds': builtins.round(float(total_time) / stocks_count, 2),
            'predictions_per_second': builtins.round(stocks_count / float(total_time), 2),
            'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
            'end_time': datetime.fromtimestamp(self.end_time).isoformat()
        }
        
        return metrics
    
    def _log_performance_summary(self):
        """Log comprehensive performance summary"""
        
        metrics = self.performance_metrics
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("🎯 BATCH INFERENCE PERFORMANCE SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"📊 Stocks Processed: {metrics.get('stocks_processed', 0)}")
        self.logger.info(f"⏱️  Total Time: {metrics.get('total_inference_time_seconds', 0)} seconds ({metrics.get('total_inference_time_minutes', 0)} minutes)")
        self.logger.info(f"⚡ Average Time per Stock: {metrics.get('avg_time_per_stock_seconds', 0)} seconds")
        self.logger.info(f"🚀 Throughput: {metrics.get('predictions_per_second', 0)} predictions/second")
        self.logger.info(f"🕒 Start Time: {metrics.get('start_time', 'N/A')}")
        self.logger.info(f"🕓 End Time: {metrics.get('end_time', 'N/A')}")
        self.logger.info("=" * 80)


def create_spark_session() -> SparkSession:
    """Create Spark session for inference batch processing"""
    
    builder = SparkSession.builder \
        .appName("StockPredictionInferenceBatch") \
        .master("local[8]") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "6g") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    
    # Performance optimizations for inference
    builder = builder.config("spark.sql.adaptive.enabled", "true") \
                    .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "32MB") \
                    .config("spark.ml.cache.enabled", "true") \
                    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    
    return spark


def main():
    """Main inference batch function"""
    
    print("🚀 Starting Batch Inference for Stock Price Prediction")
    print("=" * 80)
    
    spark = None
    try:
        # Load environment and configuration
        load_environment()
        multi_config = MultiModeConfig("config/config_modes.yaml")
        
        print("🔧 Configuration loaded successfully")
        print(f"📊 Inference stocks: {len(multi_config._config['inference_modes']['spark_rf_inference']['inference_stocks'])}")
        
        # Create Spark session
        spark = create_spark_session()
        print("🔧 Spark session created successfully")
        
        # Create inference processor
        processor = InferenceBatchProcessor(multi_config, spark)
        
        # Run batch inference
        results = processor.run_batch_inference()
        
        # Print summary
        print("\n" + "=" * 80)
        print("🎉 Batch Inference Completed Successfully!")
        print("=" * 80)
        print(f"✅ Status: {results['status']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Batch inference failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        if spark is not None:
            try:
                spark.stop()
                print("🔧 Spark session stopped")
            except:
                pass


def run_inference_batch():
    """
    Wrapper function for Airflow DAG integration.
    Runs the complete inference batch pipeline.
    """
    print("[INFERENCE] Starting batch inference task...")
    
    try:
        results = main()
        print(f"[INFERENCE] ✅ Batch inference completed successfully")
        print(f"[INFERENCE] Predictions saved to: {results.get('delta_table_path', 'N/A')}")
        return results
    except Exception as e:
        print(f"[INFERENCE] ❌ Batch inference failed: {e}")
        raise


if __name__ == "__main__":
    main()