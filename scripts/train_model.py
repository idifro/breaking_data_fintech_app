#!/usr/bin/env python3
"""
Training Script for Spark ML Stock Price Prediction Pipeline
Main entry point for model training with hyperparameter tuning

CONDA ENVIRONMENT: breaking_data
Usage: 
    conda activate breaking_data
    python scripts/train_model.py
"""

import sys
import os
from pathlib import Path

# Add src to path


# Add the parent directory (pipeline root) to Python path
pipeline_root = Path(__file__).parent.parent
sys.path.insert(0, str(pipeline_root))

# sys.path.append(str(Path(__file__).parent.parent / "src"))

import warnings
warnings.filterwarnings("ignore")

from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip

# Local imports
from src.utils import Config, Logger, MLflowManager, PathManager, load_environment
from src.data_processing import DataProcessor
from src.model_training import ModelTrainer


def create_spark_session(config: Config) -> SparkSession:
    """Create Spark session with configuration"""
    
    builder = SparkSession.builder \
        .appName(config.spark.app_name) \
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
    """Main training pipeline"""
    
    print("🚀 Starting Spark ML Stock Price Prediction Training Pipeline")
    print("=" * 70)
    
    try:
        # Load environment variables
        load_environment()
        
        # Load configuration
        config = Config()
        
        # Setup logging
        logger = Logger.setup_logging(config)
        logger.info("Starting training pipeline...")
        
        # Initialize path manager
        path_manager = PathManager()
        
        # Create Spark session
        logger.info("Creating Spark session...")
        spark = create_spark_session(config)
        logger.info(f"Spark session created: {spark.sparkContext.appName}")
        
        # Print Spark cluster and MLflow URLs explicitly
        print("\n" + "=" * 70)
        print("🔗 CLUSTER AND TRACKING INFORMATION")
        print("=" * 70)
        print(f"🌟 Spark Master URL: {spark.sparkContext.master}")
        print(f"📊 Spark Application ID: {spark.sparkContext.applicationId}")
        print(f"🎯 Spark UI URL: {spark.sparkContext.uiWebUrl}")
        
        # Setup MLflow and print tracking info
        mlflow_manager = MLflowManager(config)
        import mlflow
        mlflow_tracking_uri = mlflow.get_tracking_uri()
        print(f"🔬 MLflow Tracking URI: {mlflow_tracking_uri}")
        print(f"🧪 MLflow Experiment: {config.mlflow_config['experiment_name']}")
        if mlflow_tracking_uri.startswith('file://'):
            mlflow_ui_path = mlflow_tracking_uri.replace('file://', '')
            print(f"💡 MLflow UI Command: mlflow ui --backend-store-uri {mlflow_tracking_uri}")
        print("=" * 70 + "\n")
        
        # Initialize components
        data_processor = DataProcessor(config, spark)
        model_trainer = ModelTrainer(config, spark)
        
        # Step 1: Load CSV data to Delta tables (if not already done)
        logger.info("Step 1: Loading CSV data to Delta tables...")
        try:
            # Try to load existing data
            df = data_processor.load_data_from_delta()
            logger.info("✅ Delta table already exists, using existing data")
        except FileNotFoundError:
            # If no Delta table exists, create one from CSV files
            logger.info("Delta table not found, loading from CSV files...")
            data_processor.load_csv_to_delta()
            df = data_processor.load_data_from_delta()
            logger.info("✅ Data loaded to Delta table successfully")
        
        # Step 2: Prepare training data with features
        logger.info("Step 2: Preparing training data with features...")
        train_df, test_df, feature_names = data_processor.prepare_training_data(df)
        logger.info("✅ Training data prepared successfully")
        
        # Print data summary
        logger.info(f"Training samples: {train_df.count()}")
        logger.info(f"Test samples: {test_df.count()}")
        logger.info(f"Number of features: {len(feature_names)}")
        
        # Step 3: Train model with hyperparameter tuning
        if config.hyperparameter_tuning['enabled']:
            tuning_method = config.hyperparameter_tuning['method'].upper()
            hypertuning_status = f"ENABLED ({tuning_method})"
        else:
            hypertuning_status = "DISABLED"
        logger.info(f"Step 3: Training model - Hyperparameter tuning: {hypertuning_status}")
        model, evaluation_results = model_trainer.train_model(train_df, test_df, feature_names)
        logger.info("✅ Model training completed successfully")
        
        # Print final results
        print("\n" + "=" * 70)
        print("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        # Print summary results
        test_metrics = evaluation_results['test_metrics']
        print(f"\n📊 Final Test Results:")
        print(f"   MAE: {test_metrics['mae']:.6f}")
        print(f"   MSE: {test_metrics['mse']:.6f}")
        print(f"   RMSE: {test_metrics['rmse']:.6f}")
        print(f"   R²: {test_metrics['r2']:.4f}")
        
        print(f"\n📁 Results saved to:")
        print(f"   - Model artifacts: MLflow")
        print(f"   - Evaluation results: results/training/")
        print(f"   - Visualizations: plots/")
        
        print(f"\n🔬 View MLflow UI:")
        if mlflow_tracking_uri.startswith('file://'):
            mlflow_local_path = mlflow_tracking_uri.replace('file://', '')
            print(f"   Local UI Command: mlflow ui --backend-store-uri {mlflow_tracking_uri}")
            print(f"   Then open: http://localhost:5000")
        else:
            print(f"   MLflow UI URL: {mlflow_tracking_uri}")
        
        print(f"\n🌐 Spark Cluster Information:")
        print(f"   Cluster Master: {config.spark.master}")
        print(f"   Driver Memory: {config.spark.driver_memory}")
        print(f"   Executor Memory: {config.spark.executor_memory}")
        if spark.sparkContext.uiWebUrl:
            print(f"   Spark Web UI: {spark.sparkContext.uiWebUrl}")
        else:
            print(f"   Spark Web UI: http://localhost:4040")
        
        print(f"\n📈 Top 5 Important Features:")
        for idx, row in evaluation_results['feature_importance'].head(5).iterrows():
            print(f"   {idx+1}. {row['feature']}: {row['importance']:.4f}")
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Training pipeline failed: {str(e)}")
        logger.error(f"Training pipeline failed: {str(e)}")
        raise
    
    finally:
        # Clean up Spark session
        if 'spark' in locals():
            spark.stop()
            logger.info("Spark session stopped")


if __name__ == "__main__":
    main()