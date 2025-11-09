#!/usr/bin/env python3
"""
Inference Script for Spark ML Stock Price Prediction Pipeline
Loads trained model and makes predictions on latest data
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from delta import configure_spark_with_delta_pip
import mlflow
import mlflow.spark

# Local imports
from src.utils import Config, Logger, MLflowManager, PathManager, load_environment
from src.data_processing import DataProcessor
from src.feature_engineering import FeatureEngineer


class InferencePipeline:
    """Inference pipeline for stock price prediction"""
    
    def __init__(self, config: Config, spark: SparkSession):
        self.config = config
        self.spark = spark
        self.path_manager = PathManager()
        self.data_processor = DataProcessor(config, spark)
        self.feature_engineer = FeatureEngineer(config, spark)
        self.mlflow_manager = MLflowManager(config)
    
    def load_latest_model(self, model_name: str = None):
        """Load the latest trained model from MLflow"""
        
        if model_name is None:
            model_name = self.config.mlflow_config['model_name']
        
        print(f"🔍 Loading latest model: {model_name}")
        
        try:
            # Try to load from Production stage first
            model_uri = f"models:/{model_name}/Production"
            model = mlflow.spark.load_model(model_uri)
            print(f"✅ Loaded model from Production stage")
            return model, "Production"
            
        except Exception:
            try:
                # If no Production model, try latest version
                client = mlflow.tracking.MlflowClient()
                latest_version = client.get_latest_versions(model_name, stages=["None"])
                
                if latest_version:
                    model_uri = f"models:/{model_name}/{latest_version[0].version}"
                    model = mlflow.spark.load_model(model_uri)
                    print(f"✅ Loaded model version: {latest_version[0].version}")
                    return model, latest_version[0].version
                else:
                    raise Exception(f"No model found with name: {model_name}")
                    
            except Exception as e:
                # Last resort: load from latest run
                print("⚠️ No registered model found, trying to load from latest run...")
                experiment = mlflow.get_experiment_by_name(self.config.mlflow_config['experiment_name'])
                
                if experiment:
                    runs = mlflow.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        order_by=["start_time DESC"],
                        max_results=1
                    )
                    
                    if not runs.empty:
                        run_id = runs.iloc[0]['run_id']
                        model_uri = f"runs:/{run_id}/model"
                        model = mlflow.spark.load_model(model_uri)
                        print(f"✅ Loaded model from run: {run_id}")
                        return model, run_id
                
                raise Exception(f"Could not load any model: {e}")
    
    def prepare_inference_data(self):
        """Prepare latest data for inference"""
        
        print("📊 Preparing inference data...")
        
        # Load latest data from Delta table
        df = self.data_processor.load_data_from_delta()
        
        # Get the latest sequence_length rows for each stock
        sequence_length = self.config.data.sequence_length
        
        window_spec = Window.partitionBy("stock_symbol").orderBy(desc("Date"))
        latest_df = df.withColumn("row_num", row_number().over(window_spec)) \
                     .filter(col("row_num") <= sequence_length) \
                     .drop("row_num") \
                     .orderBy("stock_symbol", "Date")
        
        # Create features using the same feature engineering pipeline
        df_with_features, feature_names = self.feature_engineer.create_all_features(latest_df)
        
        # Filter out rows with null features (we need complete feature vectors for inference)
        for feature_name in feature_names:
            df_with_features = df_with_features.filter(col(feature_name).isNotNull())
        
        # Load scaler model
        scaler_model = self.data_processor.load_scaler_model()
        
        # Create feature vectors
        from pyspark.ml.feature import VectorAssembler
        assembler = VectorAssembler(inputCols=feature_names, outputCol="features_raw")
        df_features = assembler.transform(df_with_features)
        
        # Scale features
        df_scaled = scaler_model.transform(df_features)
        
        print(f"✅ Inference data prepared for {df_scaled.select('stock_symbol').distinct().count()} stocks")
        
        return df_scaled, feature_names
    
    def make_predictions(self, model, inference_df, feature_names: List[str]):
        """Make predictions using the trained model"""
        
        print("🔮 Making predictions...")
        
        # Make predictions
        predictions_df = model.transform(inference_df)
        
        # Collect results
        results = predictions_df.select(
            "stock_symbol", 
            "Date", 
            "Close", 
            "prediction",
            "features"
        ).collect()
        
        # Convert to more readable format
        predictions = []
        for row in results:
            # Get the most recent prediction for each stock
            predictions.append({
                'stock_symbol': row.stock_symbol,
                'current_date': row.Date,
                'current_price': row.Close,
                'predicted_return': row.prediction,
                'predicted_price': row.Close * (1 + row.prediction),
                'timestamp': pd.Timestamp.now()
            })
        
        # Get latest prediction for each stock (most recent date)
        predictions_df_pandas = pd.DataFrame(predictions)
        if not predictions_df_pandas.empty:
            latest_predictions = predictions_df_pandas.groupby('stock_symbol').apply(
                lambda x: x.loc[x['current_date'].idxmax()]
            ).reset_index(drop=True)
        else:
            latest_predictions = predictions_df_pandas
        
        print(f"✅ Predictions made for {len(latest_predictions)} stocks")
        
        return latest_predictions
    
    def save_predictions(self, predictions_df: pd.DataFrame):
        """Save predictions to Delta table and CSV"""
        
        print("💾 Saving predictions...")
        
        if predictions_df.empty:
            print("⚠️ No predictions to save")
            return
        
        # Convert to Spark DataFrame
        predictions_spark = self.spark.createDataFrame(predictions_df)
        
        # Save to Delta table
        delta_path = str(self.path_manager.delta_tables_dir / "predictions")
        predictions_spark.write.format("delta") \
                         .mode("append") \
                         .save(delta_path)
        
        # Save to CSV with timestamp
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.path_manager.results_dir / "inference" / f"predictions_{timestamp}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        predictions_df.to_csv(csv_path, index=False)
        
        print(f"✅ Predictions saved to:")
        print(f"   - Delta table: {delta_path}")
        print(f"   - CSV file: {csv_path}")
    
    def print_predictions(self, predictions_df: pd.DataFrame):
        """Print predictions in a formatted way"""
        
        if predictions_df.empty:
            print("⚠️ No predictions available")
            return
        
        print("\n" + "=" * 80)
        print("📈 STOCK PRICE PREDICTIONS")
        print("=" * 80)
        
        for _, row in predictions_df.iterrows():
            predicted_change = (row['predicted_price'] - row['current_price']) / row['current_price']
            direction = "📈" if predicted_change > 0 else "📉" if predicted_change < 0 else "➡️"
            
            print(f"\n{direction} {row['stock_symbol']}:")
            print(f"   Current Price: ${row['current_price']:.2f}")
            print(f"   Predicted Price: ${row['predicted_price']:.2f}")
            print(f"   Predicted Return: {row['predicted_return']:.4f} ({predicted_change:.2%})")
            print(f"   Date: {row['current_date']}")
        
        print("\n" + "=" * 80)
        
        # Summary statistics
        avg_return = predictions_df['predicted_return'].mean()
        max_return = predictions_df['predicted_return'].max()
        min_return = predictions_df['predicted_return'].min()
        
        print(f"📊 PREDICTION SUMMARY:")
        print(f"   Average Predicted Return: {avg_return:.4f} ({avg_return:.2%})")
        print(f"   Highest Predicted Return: {max_return:.4f} ({max_return:.2%})")
        print(f"   Lowest Predicted Return: {min_return:.4f} ({min_return:.2%})")
        print(f"   Bullish Stocks: {len(predictions_df[predictions_df['predicted_return'] > 0])}")
        print(f"   Bearish Stocks: {len(predictions_df[predictions_df['predicted_return'] < 0])}")
    
    def run_inference(self):
        """Run complete inference pipeline"""
        
        try:
            # Load model
            model, model_info = self.load_latest_model()
            
            # Prepare data
            inference_df, feature_names = self.prepare_inference_data()
            
            # Make predictions
            predictions_df = self.make_predictions(model, inference_df, feature_names)
            
            # Print results
            self.print_predictions(predictions_df)
            
            # Save results
            self.save_predictions(predictions_df)
            
            return predictions_df
            
        except Exception as e:
            print(f"❌ Inference failed: {str(e)}")
            raise


def create_spark_session(config: Config) -> SparkSession:
    """Create Spark session for inference"""
    
    builder = SparkSession.builder \
        .appName("StockPredictionInference") \
        .master(config.spark.master) \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    
    return configure_spark_with_delta_pip(builder).getOrCreate()


def main():
    """Main inference function"""
    
    print("🔮 Starting Stock Price Prediction Inference Pipeline")
    print("=" * 60)
    
    try:
        # Load environment and config
        load_environment()
        config = Config()
        
        # Setup logging
        logger = Logger.setup_logging(config)
        logger.info("Starting inference pipeline...")
        
        # Create Spark session
        spark = create_spark_session(config)
        
        # Initialize inference pipeline
        inference_pipeline = InferencePipeline(config, spark)
        
        # Run inference
        predictions_df = inference_pipeline.run_inference()
        
        print("\n🎉 Inference completed successfully!")
        logger.info("Inference pipeline completed successfully")
        
        return predictions_df
        
    except Exception as e:
        print(f"\n❌ Inference pipeline failed: {str(e)}")
        logger.error(f"Inference pipeline failed: {str(e)}")
        raise
    
    finally:
        # Clean up
        if 'spark' in locals():
            spark.stop()


if __name__ == "__main__":
    main()