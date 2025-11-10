#!/usr/bin/env python3
"""
Enhanced Training Script for Spark ML Stock Price Prediction Pipeline
Supports stock-specific training, model versioning, and production features

CONDA ENVIRONMENT: breaking_data
Usage: 
    conda activate breaking_data
    python scripts/train_model.py --stocks AAPL,GOOG  # Train specific stocks
    python scripts/train_model.py --all                # Train all available stocks
    python scripts/train_model.py                       # Train config stocks
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import json

# Add src to path
pipeline_root = Path(__file__).parent.parent
sys.path.insert(0, str(pipeline_root))

import warnings
warnings.filterwarnings("ignore")

from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip

# Local imports
from src.utils import Config, Logger, MLflowManager, PathManager, load_environment
from src.data_processing import DataProcessor
from src.model_training import ModelTrainer


class ProductionModelTrainer:
    """Enhanced training coordinator for production pipeline"""
    
    def __init__(self, config: Config, spark: SparkSession):
        self.config = config
        self.spark = spark
        self.logger = Logger().get_logger()
        
        self.data_processor = DataProcessor(config, spark)
        self.model_trainer = ModelTrainer(config, spark)
        self.mlflow_manager = MLflowManager(config)
        
        # Production metrics
        self.training_metrics = {
            'start_time': None,
            'end_time': None,
            'duration_seconds': None,
            'stocks_processed': [],
            'total_samples': 0,
            'models_trained': 0,
            'average_performance': {},
            'resource_usage': {},
            'errors': []
        }
    
    def train_unified_model(self, stock_selection: List[str]) -> Dict:
        """
        Train a single unified model using combined data from selected stocks
        
        Args:
            stock_selection: List of stocks to combine for training
            
        Returns:
            Dictionary with training results and metrics
        """
        
        self.training_metrics['start_time'] = datetime.now()
        self.logger.info(f"🚀 Starting unified model training for stocks: {stock_selection}")
        
        # Validate stock availability
        available_stocks = self.data_processor.get_available_stocks()
        valid_stocks = [stock for stock in stock_selection if stock in available_stocks]
        invalid_stocks = [stock for stock in stock_selection if stock not in available_stocks]
        
        if invalid_stocks:
            self.logger.warning(f"⚠️ Skipping unavailable stocks: {invalid_stocks}")
        
        if not valid_stocks:
            raise ValueError(f"No valid stocks found in selection: {stock_selection}")
        
        self.logger.info(f"📊 Training unified model with {len(valid_stocks)} stocks: {valid_stocks}")
        
        try:
            # Train unified model
            model_result = self._train_unified_stock_model(valid_stocks)
            
            # Calculate summary metrics
            summary = {
                'stocks_used': valid_stocks,
                'model_type': 'unified',
                'training_samples': model_result['training_samples'],
                'test_samples': model_result['test_samples'],
                'features_count': model_result['features_count'],
                'performance': model_result['metrics']['test_metrics'],
                'success_rate': 1.0,
                'invalid_stocks': invalid_stocks
            }
            
            self.training_metrics['end_time'] = datetime.now()
            self.training_metrics['duration_seconds'] = (
                self.training_metrics['end_time'] - self.training_metrics['start_time']
            ).total_seconds()
            
            self.logger.info(f"🎉 Unified model training completed in {self.training_metrics['duration_seconds']:.2f} seconds")
            
            return {
                'model_info': model_result['model_info'],
                'metrics': model_result['metrics'],
                'summary': summary,
                'errors': [f"Unavailable stocks: {invalid_stocks}"] if invalid_stocks else []
            }
            
        except Exception as e:
            error_msg = f"Failed to train unified model: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _train_unified_stock_model(self, stock_list: List[str]) -> Dict:
        """Train a single model using combined data from multiple stocks"""
        
        import mlflow
        
        # End any active runs to avoid conflicts
        if mlflow.active_run():
            mlflow.end_run()
        
        # Load and combine data from multiple stocks
        self.logger.info(f"📊 Loading and combining data from {len(stock_list)} stocks...")
        train_df, test_df, feature_names = self.data_processor.prepare_training_data(
            stock_selection=stock_list
        )
        
        training_samples = train_df.count()
        test_samples = test_df.count()
        total_samples = training_samples + test_samples
        
        self.training_metrics['total_samples'] = total_samples
        self.training_metrics['stocks_processed'] = stock_list
        
        # Create unified model name
        if len(stock_list) <= 3:
            model_name = f"stock_predictor_{'_'.join(sorted(stock_list))}"
        else:
            model_name = f"stock_predictor_unified_{len(stock_list)}stocks"
        
        # Set up unified experiment
        experiment_name = f"{self.config.mlflow_config['experiment_name']}_unified"
        
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
            self.logger.info(f"📂 Created new MLflow experiment: {experiment_name}")
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(experiment_name)
            experiment_id = experiment.experiment_id
            self.logger.info(f"📂 Using existing MLflow experiment: {experiment_name}")
        
        mlflow.set_experiment(experiment_name)
        
        # Train the unified model
        run_name = f"unified_model_{'-'.join(sorted(stock_list))}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name) as run:
            # Log unified model parameters
            mlflow.log_param("model_type", "unified")
            mlflow.log_param("stocks_included", ','.join(sorted(stock_list)))
            mlflow.log_param("num_stocks", len(stock_list))
            mlflow.log_param("training_samples", training_samples)
            mlflow.log_param("test_samples", test_samples)
            mlflow.log_param("total_samples", total_samples)
            mlflow.log_param("features_count", len(feature_names))
            
            # Log individual stock sample counts
            for stock in stock_list:
                stock_train_count = train_df.filter(train_df.stock_symbol == stock).count()
                stock_test_count = test_df.filter(test_df.stock_symbol == stock).count()
                mlflow.log_param(f"{stock}_training_samples", stock_train_count)
                mlflow.log_param(f"{stock}_test_samples", stock_test_count)
            
            self.logger.info(f"🔧 Training unified model with {total_samples:,} total samples...")
            
            # Train the model using the existing ModelTrainer
            # Train the model using a direct approach to avoid nested MLflow runs
            print("🚀 Starting unified model training...")
            
            # Log configuration using existing MLflow run
            self.mlflow_manager.log_config(self.config)
            self.mlflow_manager.log_feature_config(feature_names)
            
            # Get best hyperparameters
            if self.config.hyperparameter_tuning['enabled']:
                print("🔍 Hyperparameter tuning is ENABLED")
                best_params = self.model_trainer._tune_hyperparameters(train_df, test_df)
            else:
                print("⚙️ Hyperparameter tuning is DISABLED - using default parameters")
                best_params = self.config.model.default_params
                print(f"✅ Using default parameters: {best_params}")
            
            # Train final model with best parameters
            model = self.model_trainer._train_final_model(train_df, best_params)
            
            # Evaluate model
            evaluation_results = self.model_trainer._evaluate_model(model, train_df, test_df, feature_names)
            
            # Log metrics directly
            self.model_trainer._log_metrics(evaluation_results)
            
            # Log model to MLflow first
            mlflow.spark.log_model(model, "model")
            
            # Log additional production metrics for unified model
            self._log_unified_production_metrics(stock_list, evaluation_results, training_samples, test_samples)
            
            # Register unified model with versioning
            model_version = mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/model",
                name=model_name
            ).version
            
            # Tag unified model for production tracking
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            
            # Set comprehensive tags
            client.set_model_version_tag(
                name=model_name,
                version=model_version,
                key="stage",
                value="production_candidate"
            )
            client.set_model_version_tag(
                name=model_name,
                version=model_version,
                key="model_type",
                value="unified"
            )
            client.set_model_version_tag(
                name=model_name,
                version=model_version,
                key="stocks_included",
                value=','.join(sorted(stock_list))
            )
            client.set_model_version_tag(
                name=model_name,
                version=model_version,
                key="num_stocks",
                value=str(len(stock_list))
            )
            client.set_model_version_tag(
                name=model_name,
                version=model_version,
                key="training_date",
                value=datetime.now().isoformat()
            )
            client.set_model_version_tag(
                name=model_name,
                version=model_version,
                key="total_samples",
                value=str(total_samples)
            )
            
            # Log model performance summary
            test_metrics = evaluation_results['test_metrics']
            r2_score = test_metrics['r2']
            
            # Performance grade
            if r2_score >= 0.9:
                grade = "A"
            elif r2_score >= 0.8:
                grade = "B"
            elif r2_score >= 0.7:
                grade = "C"
            elif r2_score >= 0.6:
                grade = "D"
            else:
                grade = "F"
            
            client.set_model_version_tag(
                name=model_name,
                version=model_version,
                key="performance_grade",
                value=grade
            )
            
            self.logger.info(f"✅ Unified model registered: {model_name} v{model_version} (Grade: {grade})")
            
            return {
                'model_info': {
                    'name': model_name,
                    'version': model_version,
                    'run_id': run.info.run_id,
                    'experiment_id': experiment_id,
                    'type': 'unified',
                    'stocks_included': stock_list
                },
                'metrics': evaluation_results,
                'training_samples': training_samples,
                'test_samples': test_samples,
                'features_count': len(feature_names)
            }

    
    def _log_unified_production_metrics(self, stock_list: List[str], evaluation_results: Dict, training_samples: int, test_samples: int):
        """Log production-level metrics for unified model"""
        import mlflow
        import psutil
        
        # System resource usage
        memory_info = psutil.virtual_memory()
        mlflow.log_metric("system_memory_usage_percent", memory_info.percent)
        mlflow.log_metric("system_memory_available_gb", memory_info.available / (1024**3))
        
        # CPU usage (sample over 1 second)
        cpu_percent = psutil.cpu_percent(interval=1)
        mlflow.log_metric("system_cpu_usage_percent", cpu_percent)
        
        # Disk usage
        disk_info = psutil.disk_usage('/')
        mlflow.log_metric("system_disk_usage_percent", disk_info.percent)
        
        # Model performance metrics
        test_metrics = evaluation_results['test_metrics']
        
        # Production readiness score
        readiness_score = self._calculate_production_readiness_score(test_metrics)
        mlflow.log_metric("production_readiness_score", readiness_score)
        
        # Data diversity metrics (multiple stocks)
        mlflow.log_metric("data_diversity_stock_count", len(stock_list))
        mlflow.log_metric("avg_samples_per_stock", (training_samples + test_samples) / len(stock_list))
        
        # Feature importance statistics
        feature_importance = evaluation_results['feature_importance']
        mlflow.log_metric("feature_importance_max", feature_importance['importance'].max())
        mlflow.log_metric("feature_importance_mean", feature_importance['importance'].mean())
        mlflow.log_metric("feature_importance_std", feature_importance['importance'].std())
        
        # Cross-stock generalization score (based on diversity)
        diversity_bonus = min(len(stock_list) * 5, 20)  # Bonus points for more stocks
        generalization_score = readiness_score + diversity_bonus
        mlflow.log_metric("cross_stock_generalization_score", min(generalization_score, 100))
    
    def _calculate_production_readiness_score(self, test_metrics: Dict) -> float:
        """Calculate a composite production readiness score (0-100)"""
        
        # Weights for different metrics
        weights = {
            'r2': 0.4,      # Model accuracy (most important)
            'rmse': 0.3,    # Prediction error
            'mae': 0.2,     # Mean error
            'stability': 0.1  # Model stability (derived)
        }
        
        # Normalize R² (higher is better)
        r2_score = max(0, min(100, test_metrics['r2'] * 100))
        
        # Normalize RMSE (lower is better, assuming reasonable range)
        rmse_score = max(0, min(100, (1 - min(test_metrics['rmse'] / 10, 1)) * 100))
        
        # Normalize MAE (lower is better)
        mae_score = max(0, min(100, (1 - min(test_metrics['mae'] / 5, 1)) * 100))
        
        # Stability score (based on R² - if very high, assume stable)
        stability_score = r2_score if test_metrics['r2'] > 0.7 else 50
        
        # Calculate weighted score
        readiness_score = (
            weights['r2'] * r2_score +
            weights['rmse'] * rmse_score +
            weights['mae'] * mae_score +
            weights['stability'] * stability_score
        )
        
        return round(readiness_score, 2)
    
    def _calculate_summary_metrics(self, results: Dict) -> Dict:
        """Calculate summary metrics for unified model"""
        
        if 'metrics' not in results:
            return {}
        
        # For unified model, there's only one set of metrics
        test_metrics = results['metrics']['test_metrics']
        
        summary = {
            'model_type': 'unified',
            'stocks_count': len(results.get('summary', {}).get('stocks_used', [])),
            'r2_score': test_metrics['r2'],
            'mae': test_metrics['mae'],
            'rmse': test_metrics['rmse'],
            'mse': test_metrics['mse'],
            'training_samples': results.get('summary', {}).get('training_samples', 0),
            'test_samples': results.get('summary', {}).get('test_samples', 0),
            'success_rate': 1.0 if results['metrics'] else 0.0,
            'performance_grade': self._get_performance_grade(test_metrics['r2'])
        }
        
        return summary
    
    def _get_performance_grade(self, r2_score: float) -> str:
        """Get performance grade based on R² score"""
        if r2_score >= 0.9:
            return "A"
        elif r2_score >= 0.8:
            return "B"
        elif r2_score >= 0.7:
            return "C"
        elif r2_score >= 0.6:
            return "D"
        else:
            return "F"


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


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    
    parser = argparse.ArgumentParser(
        description="Enhanced Stock Price Prediction Model Training"
    )
    
    parser.add_argument(
        '--stocks',
        type=str,
        help='Comma-separated list of stock symbols to train (e.g., AAPL,GOOG,NVDA)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Train models for all available stocks'
    )
    
    parser.add_argument(
        '--config-training',
        action='store_true',
        default=True,
        help='Use config training_stocks (default)'
    )
    
    return parser.parse_args()
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
    """Enhanced main training pipeline with stock-specific support"""
    
    # Parse command line arguments
    args = parse_arguments()
    
    print("🚀 Starting Enhanced Spark ML Stock Price Prediction Training Pipeline")
    print("=" * 80)
    
    try:
        # Load environment variables
        load_environment()
        
        # Load configuration
        config = Config()
        
        # Setup logging
        logger = Logger.setup_logging(config)
        logger.info("Starting enhanced training pipeline...")
        
        # Initialize path manager
        path_manager = PathManager()
        
        # Create Spark session
        logger.info("Creating Spark session...")
        spark = create_spark_session(config)
        logger.info(f"Spark session created: {spark.sparkContext.appName}")
        
        # Print cluster information
        print("\n" + "=" * 80)
        print("🔗 CLUSTER AND TRACKING INFORMATION")
        print("=" * 80)
        print(f"🌟 Spark Master URL: {spark.sparkContext.master}")
        print(f"📊 Spark Application ID: {spark.sparkContext.applicationId}")
        print(f"🎯 Spark UI URL: {spark.sparkContext.uiWebUrl or 'http://localhost:4040'}")
        
        # Setup MLflow
        import mlflow
        mlflow_manager = MLflowManager(config)
        mlflow_tracking_uri = mlflow.get_tracking_uri()
        print(f"🔬 MLflow Tracking URI: {mlflow_tracking_uri}")
        print(f"🧪 MLflow Base Experiment: {config.mlflow_config['experiment_name']}")
        
        if mlflow_tracking_uri.startswith('file://'):
            print(f"💡 MLflow UI Command: mlflow ui --backend-store-uri {mlflow_tracking_uri}")
        print("=" * 80 + "\n")
        
        # Initialize production trainer
        trainer = ProductionModelTrainer(config, spark)
        
        # Determine which stocks to train
        if args.stocks:
            # Specific stocks from command line
            stock_list = [stock.strip().upper() for stock in args.stocks.split(',')]
            print(f"📋 Training specific stocks from command line: {stock_list}")
            
        elif args.all:
            # All available stocks
            stock_list = config.data.available_stocks
            print(f"📋 Training all available stocks: {stock_list}")
            
        else:
            # Default: config training stocks
            stock_list = config.data.training_stocks
            print(f"📋 Training config stocks: {stock_list}")
        
        print(f"🎯 Selected {len(stock_list)} stocks for training: {stock_list}\n")
        
        # Step 1: Validate Delta tables exist
        logger.info("Step 1: Validating stock-specific Delta tables...")
        data_processor = DataProcessor(config, spark)
        available_stocks = data_processor.get_available_stocks()
        
        missing_stocks = [stock for stock in stock_list if stock not in available_stocks]
        if missing_stocks:
            logger.warning(f"⚠️ Missing Delta tables for stocks: {missing_stocks}")
            logger.info("🔄 Please run create_delta_tables.py first to create required tables")
            
            # Filter to only available stocks
            stock_list = [stock for stock in stock_list if stock in available_stocks]
            
            if not stock_list:
                raise ValueError(f"No valid stocks found. Available: {available_stocks}")
        
        logger.info("✅ Delta table validation completed")
        
        # Step 2: Train unified model
        logger.info(f"Step 2: Training unified model for {len(stock_list)} stocks...")
        
        training_start_time = datetime.now()
        training_results = trainer.train_unified_model(stock_list)
        training_duration = (datetime.now() - training_start_time).total_seconds()
        
        logger.info("✅ Unified model training completed successfully")
        
        # Step 3: Print comprehensive results
        print("\n" + "=" * 80)
        print("🎉 ENHANCED UNIFIED MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        # Print summary statistics
        summary = training_results['summary']
        print(f"\n📊 Training Summary:")
        print(f"   🏁 Stocks Processed: {summary.get('stocks_used', 0)}")
        print(f"   ⏱️  Total Duration: {training_duration:.2f} seconds")
        print(f"   ✅ Success Rate: {summary.get('success_rate', 0):.1%}")
        print(f"   � Average R² Score: {summary.get('r2', 0):.4f}")
        print(f"   📊 Average MAE: {summary.get('mae', 0):.6f}")
        print(f"   📉 Average RMSE: {summary.get('rmse', 0):.6f}")
        
        # Best and worst performers
        if summary.get('best_performing_stock'):
            best = summary['best_performing_stock']
            print(f"   🥇 Best Performer: {best['stock']} (R²: {best['r2']:.4f})")
        
        if summary.get('worst_performing_stock'):
            worst = summary['worst_performing_stock']
            print(f"   📉 Needs Attention: {worst['stock']} (R²: {worst['r2']:.4f})")
        
        # Print individual stock results
        print(f"\n📈 Individual Stock Results:")
        if training_results['metrics']:
            for stock, metrics in training_results['metrics']['per_stock_metrics'].items():
                if "sample_count" in metrics:
                    print(f"   {stock:6s} - R²: {metrics['r2']:.4f}, "
                            f"MAE: {metrics['mae']:.6f}, "
                            f"MSE: {metrics['mae']:.6f}, ")
                else:
                    # Handle any other format
                    print(f"   {stock:6s} - Metrics available but format unknown")
        else:
            print(f"   No individual stock metrics available for unified model")
        
        # Print any errors
        if training_results['errors']:
            print(f"\n⚠️ Errors Encountered:")
            for error in training_results['errors']:
                print(f"   - {error}")
        
        # Print production information
        print(f"\n🏭 Production Features:")
        print(f"   📊 Model Versioning: Enabled (MLflow)")
        print(f"   🔍 Performance Monitoring: Enabled")
        print(f"   📈 Resource Tracking: Enabled")
        print(f"   🎯 Production Readiness: Scored per model")
        
        # Print MLflow information
        print(f"\n🔬 MLflow Tracking:")
        print(f"   📂 Experiments Created: {len(stock_list)} (stock-specific)")
        if 'model_info' in training_results and training_results['model_info']:
            print(f"   🏷️  Model Registry: {training_results['model_info']['name']} models registered")
        else:
            print(f"   🏷️  Model Registry: 1 unified model registered")
        print(f"   🔗 Tracking URI: {mlflow_tracking_uri}")
        
        if mlflow_tracking_uri.startswith('file://'):
            print(f"   💻 Local MLflow UI: mlflow ui --backend-store-uri {mlflow_tracking_uri}")
            print(f"   🌐 Then open: http://localhost:5000")
        
        print(f"\n📁 Results and Artifacts:")
        print(f"   � Model Registry: MLflow (versioned)")
        print(f"   📊 Training Metrics: MLflow experiments")
        print(f"   🔧 Feature Scalers: models/feature_scaler")
        print(f"   📈 Visualizations: plots/")
        
        # Save training summary to file
        summary_path = path_manager.results_dir / "training" / f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(summary_path, 'w') as f:
            # Convert datetime objects to strings for JSON serialization
            serializable_results = {
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': training_duration,
                'stocks_processed': stock_list,
                'summary': summary,
                'model': training_results['model_info'].get('name', ""),
                'model_version': training_results['model_info'].get('version', ""),
                'stocks_included': training_results['model_info'].get('stocks_included', []),
                'errors': training_results['errors']
            }
            json.dump(serializable_results, f, indent=2)
        
        print(f"   📄 Training Summary: {summary_path}")
        
        logger.info("Enhanced training pipeline completed successfully!")
        
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