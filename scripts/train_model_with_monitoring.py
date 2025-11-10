#!/usr/bin/env python3
"""
Enhanced Training Script with Scalability Monitoring
Integrates comprehensive monitoring for Data Engineering at Scale project

CONDA ENVIRONMENT: breaking_data
Usage: 
    conda activate breaking_data
    
    # Train with monitoring (default)
    python scripts/train_model_with_monitoring.py
    
    # Train specific stocks with monitoring
    python scripts/train_model_with_monitoring.py --stocks AAPL,GOOG
    
    # Disable monitoring
    python scripts/train_model_with_monitoring.py --no-monitoring
    
    # Run load testing
    python scripts/train_model_with_monitoring.py --load-test
"""

import sys
import os
import argparse
import time
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
from src.scalability_monitor import ScalabilityMonitor, PerformanceBenchmark
from src.monitoring_config_manager import get_monitoring_config_manager


class MonitoredModelTrainer:
    """Enhanced training coordinator with scalability monitoring"""
    
    def __init__(self, config: Config, spark: SparkSession, enable_monitoring: bool = True):
        self.config = config
        self.spark = spark
        self.logger = Logger().get_logger()
        
        # Initialize core components
        self.data_processor = DataProcessor(config, spark)
        self.model_trainer = ModelTrainer(config, spark)
        self.mlflow_manager = MLflowManager(config)
        
        # Initialize monitoring
        self.enable_monitoring = enable_monitoring and config.monitoring.enabled
        if self.enable_monitoring:
            monitoring_config = {
                'enabled': True,
                'detailed_metrics': config.monitoring.detailed_metrics,
                'resource_monitoring_interval': config.monitoring.resource_monitoring_interval,
                'experiment_name': config.monitoring.experiment_name
            }
            self.monitor = ScalabilityMonitor(config, spark, monitoring_config)
            self.benchmark = PerformanceBenchmark(config, spark)
            self.logger.info("🔍 Scalability monitoring enabled")
        else:
            self.monitor = None
            self.benchmark = None
            self.logger.info("⚠️  Scalability monitoring disabled")
        
        # Training metrics
        self.training_metrics = {
            'start_time': None,
            'end_time': None,
            'stocks_processed': [],
            'models_trained': [],
            'total_training_time': 0,
            'data_volume_metrics': {},
            'performance_metrics': {},
            'scalability_metrics': None
        }
    
    def train_unified_model(self, stock_selection: List[str]) -> Dict:
        """Train a single unified model using combined data from selected stocks with comprehensive monitoring"""
        start_time = time.time()
        self.training_metrics['start_time'] = datetime.now()
        self.logger.info(f"� Starting unified model training with monitoring for stocks: {stock_selection}")
        
        # Start monitoring experiment
        if self.monitor:
            self.monitor.start_monitoring(f"training_unified")
            
            # Monitor data volume
            data_volume_metrics = self.monitor.monitor_data_volume(stock_selection)
            self._check_data_volume_warnings(data_volume_metrics)
        
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
            # Train unified model with monitoring
            model_result = self._train_unified_stock_model_with_monitoring(valid_stocks)
            
            # Monitor training efficiency
            if self.monitor:
                efficiency_metrics = self._calculate_training_efficiency_unified(model_result, start_time)
                self.monitor._log_metrics_to_mlflow()
            
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
            
            self.logger.info(f"🎉 Unified model training with monitoring completed in {self.training_metrics['duration_seconds']:.2f} seconds")
            
            return {
                'model_info': model_result['model_info'],
                'metrics': model_result['metrics'],
                'summary': summary,
                'total_training_time': self.training_metrics['duration_seconds'],
                'stocks_processed': valid_stocks,
                'models_trained': [model_result['model_info']['name']],
                'errors': [f"Unavailable stocks: {invalid_stocks}"] if invalid_stocks else []
            }
            
        except Exception as e:
            error_msg = f"Failed to train unified model: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        finally:
            if self.monitor:
                scalability_metrics = self.monitor.stop_monitoring()
                
                # Additional monitoring cleanup if needed
                self.logger.info("🏁 Monitoring completed and stopped")
    
    def train_models(self, stock_selection: List[str], run_load_test: bool = False) -> Dict:
        """Enhanced training with comprehensive monitoring (wrapper for unified approach)"""
        return self.train_unified_model(stock_selection)
    
    def _train_unified_stock_model_with_monitoring(self, stock_list: List[str]) -> Dict:
        """Train a single model using combined data from multiple stocks with detailed monitoring"""
        
        import mlflow
        
        # End any active runs to avoid conflicts
        if mlflow.active_run():
            mlflow.end_run()
        
        # Monitor data loading
        data_load_start = time.time()
        self.logger.info(f"📊 Loading and combining data from {len(stock_list)} stocks with monitoring...")
        
        train_df, test_df, feature_names = self.data_processor.prepare_training_data(
            stock_selection=stock_list
        )
        
        training_samples = train_df.count()
        test_samples = test_df.count()
        total_samples = training_samples + test_samples
        
        self.training_metrics['total_samples'] = total_samples
        self.training_metrics['stocks_processed'] = stock_list
        
        # Monitor data loading performance
        if self.monitor:
            data_load_performance = self.monitor.monitor_performance(
                "data_loading", data_load_start, total_samples
            )
            self.logger.info(f"📊 Data loading completed: {data_load_performance['duration_seconds']:.2f}s for {total_samples:,} samples")
        
        # Monitor feature engineering performance
        if self.monitor:
            feature_eng_performance = self.monitor.monitor_performance(
                "feature_engineering", data_load_start, len(feature_names)
            )
            self.logger.info(f"🔧 Feature engineering: {len(feature_names)} features created")
        
        # Create unified model name
        if len(stock_list) <= 3:
            model_name = f"stock_predictor_{'_'.join(sorted(stock_list))}"
        else:
            model_name = f"stock_predictor_unified_{len(stock_list)}stocks"
        
        # Set up unified experiment
        experiment_name = f"{self.config.mlflow_config['experiment_name']}_unified_monitoring"
        
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
            self.logger.info(f"📂 Created new MLflow experiment: {experiment_name}")
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(experiment_name)
            experiment_id = experiment.experiment_id
            self.logger.info(f"📂 Using existing MLflow experiment: {experiment_name}")
        
        mlflow.set_experiment(experiment_name)
        
        # Train the unified model with monitoring
        run_name = f"unified_model_monitoring_{'-'.join(sorted(stock_list))}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name) as run:
            # Log unified model parameters with monitoring
            mlflow.log_param("model_type", "unified_with_monitoring")
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
            
            self.logger.info(f"🔧 Training unified model with monitoring - {total_samples:,} total samples...")
            
            # Monitor model training
            model_train_start = time.time()
            
            # Log configuration using existing MLflow run
            self.mlflow_manager.log_config(self.config)
            self.mlflow_manager.log_feature_config(feature_names)
            
            # Get best hyperparameters with monitoring
            if self.config.hyperparameter_tuning['enabled']:
                self.logger.info("🔍 Hyperparameter tuning is ENABLED (with monitoring)")
                best_params = self.model_trainer._tune_hyperparameters(train_df, test_df)
            else:
                self.logger.info("⚙️ Hyperparameter tuning is DISABLED - using default parameters (with monitoring)")
                best_params = self.config.model.default_params
                self.logger.info(f"✅ Using default parameters: {best_params}")
            
            # Train final model with best parameters
            model = self.model_trainer._train_final_model(train_df, best_params)
            
            # Evaluate model
            evaluation_results = self.model_trainer._evaluate_model(model, train_df, test_df, feature_names)
            
            # Log metrics directly
            self.model_trainer._log_metrics(evaluation_results)
            
            # Monitor training performance
            if self.monitor:
                model_performance = self.monitor.monitor_performance(
                    "unified_model_training", model_train_start, total_samples
                )
                self.logger.info(f"🔧 Model training completed: {model_performance['duration_seconds']:.2f}s")
                
                # Monitor memory usage during training
                memory_usage = self.monitor.monitor_resource_usage()
                if memory_usage:
                    self.logger.info(f"💾 Peak memory usage: {memory_usage['memory_usage_mb']:.1f} MB")
                    mlflow.log_metric("peak_memory_mb", memory_usage['memory_usage_mb'])
            
            # Log model to MLflow first
            mlflow.spark.log_model(model, "model")
            
            # Register unified model with versioning
            model_version = mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/model",
                name=model_name
            ).version
            
            # Tag unified model for production tracking with monitoring
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            
            # Set comprehensive tags
            client.set_model_version_tag(
                name=model_name,
                version=model_version,
                key="stage",
                value="production_candidate_monitored"
            )
            client.set_model_version_tag(
                name=model_name,
                version=model_version,
                key="model_type",
                value="unified_monitored"
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
            
            model_path = f"models:/{model_name}/{model_version}"
            model_info = {
                'name': model_name,
                'version': model_version,
                'path': model_path,
                'type': 'unified_monitored'
            }
            
            self.logger.info(f"💾 Unified model saved: {model_path}")
            
            # Update monitoring config with successful run details
            try:
                config_manager = get_monitoring_config_manager()
                experiment_name = self.config.monitoring.experiment_name + "_training"
                config_manager.update_training_monitoring_config(
                    run.info.run_id, 
                    experiment_name
                )
                self.logger.info(f"✅ Updated training monitoring config - Run ID: {run.info.run_id}")
                self.logger.info(f"   📊 Experiment: {experiment_name}")
                self.logger.info(f"   🏃 Run Name: {run.info.run_name}")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to update training monitoring config: {e}")
            
            return {
                'model_info': model_info,
                'metrics': evaluation_results,
                'training_samples': training_samples,
                'test_samples': test_samples, 
                'features_count': len(feature_names),
                'stocks_used': stock_list,
                'model_path': model_path,
                'mlflow_run_id': run.info.run_id,
                'mlflow_experiment_name': experiment_name
            }
    
    def _check_data_volume_warnings(self, data_volume_metrics: Dict):
        """Check and warn about data volume issues"""
        total_rows = data_volume_metrics.get('total_rows', 0)
        partition_efficiency = data_volume_metrics.get('partition_efficiency', 1.0)
        
        # Check for small datasets
        if total_rows < 10000:
            self.logger.warning(f"⚠️  Small dataset detected: {total_rows:,} rows. Consider adding more data for better model performance.")
        
        # Check for partition inefficiency  
        if partition_efficiency < 0.7:
            self.logger.warning(f"⚠️  Partition inefficiency detected: {partition_efficiency:.2f}. Consider repartitioning data.")
        
        # Check memory requirements
        data_size_mb = data_volume_metrics.get('total_size_mb', 0)
        if data_size_mb > 1000:  # > 1GB
            self.logger.info(f"📊 Large dataset detected: {data_size_mb:.1f} MB. Monitoring resource usage closely.")
    
    def _calculate_training_efficiency_unified(self, model_result: Dict, start_time: float) -> Dict:
        """Calculate training efficiency metrics for unified model"""
        total_time = time.time() - start_time
        total_samples = model_result.get('training_samples', 0) + model_result.get('test_samples', 0)
        
        if not self.monitor or not self.monitor.metrics:
            return {
                'total_training_time': total_time,
                'samples_processed': total_samples,
                'throughput_samples_per_second': total_samples / total_time if total_time > 0 else 0
            }
            
        metrics = self.monitor.metrics
        
        # Calculate unified training efficiency scores
        efficiency_metrics = {
            'total_training_time': total_time,
            'samples_processed': total_samples,
            'throughput_samples_per_second': total_samples / total_time if total_time > 0 else 0,
            'data_processing_efficiency': self._calculate_data_processing_efficiency(),
            'memory_efficiency': self._calculate_memory_efficiency(),
            'time_efficiency': self._calculate_time_efficiency_unified(total_time, total_samples),
            'cost_efficiency': self._calculate_cost_efficiency()
        }
        
        # Log efficiency warnings
        for metric, value in efficiency_metrics.items():
            if isinstance(value, float) and metric.endswith('_efficiency') and value < 0.6:  # Below 60% efficiency
                self.logger.warning(f"⚠️  Low {metric}: {value:.2f}")
        
        return efficiency_metrics

    def _calculate_time_efficiency_unified(self, total_time: float, total_samples: int) -> float:
        """Calculate time efficiency for unified training"""
        if total_samples == 0 or total_time == 0:
            return 0.0
        
        # Expected baseline: ~100 samples per second for simple models
        baseline_time = total_samples / 100.0
        efficiency = min(baseline_time / total_time, 1.0) if total_time > 0 else 0.0
        
        return efficiency

    def _calculate_training_efficiency(self):
        """Calculate training efficiency metrics"""
        if not self.monitor or not self.monitor.metrics:
            return
            
        metrics = self.monitor.metrics
        
        # Calculate training efficiency scores
        training_efficiency = {
            'data_processing_efficiency': self._calculate_data_processing_efficiency(),
            'memory_efficiency': self._calculate_memory_efficiency(),
            'time_efficiency': self._calculate_time_efficiency(),
            'cost_efficiency': self._calculate_cost_efficiency()
        }
        
        self.training_metrics['training_efficiency'] = training_efficiency
        
        # Log efficiency warnings
        for metric, value in training_efficiency.items():
            if value < 0.6:  # Below 60% efficiency
                self.logger.warning(f"⚠️  Low {metric}: {value:.2f}")
    
    def _calculate_data_processing_efficiency(self) -> float:
        """Calculate data processing efficiency score"""
        # Based on throughput vs theoretical maximum
        if not self.monitor.metrics.throughput_records_per_second:
            return 0.0
            
        theoretical_max = 10000  # records per second
        actual = self.monitor.metrics.throughput_records_per_second
        
        return min(1.0, actual / theoretical_max)
    
    def _calculate_memory_efficiency(self) -> float:
        """Calculate memory efficiency score"""
        if not self.monitor.metrics.avg_memory_usage_mb:
            return 0.0
            
        # Optimal memory usage is 70-80%
        total_memory_mb = 32 * 1024  # Assume 32GB total (adjust based on system)
        usage_percent = (self.monitor.metrics.avg_memory_usage_mb / total_memory_mb) * 100
        
        if usage_percent < 50:
            return usage_percent / 70  # Underutilization
        elif usage_percent > 90:
            return (100 - usage_percent) / 10  # Overutilization
        else:
            return 1.0  # Optimal range
    
    def _calculate_time_efficiency(self) -> float:
        """Calculate time efficiency score"""
        # Based on processing time vs data volume
        total_time = self.monitor.metrics.total_processing_time
        total_rows = self.monitor.metrics.total_rows_processed
        
        if not total_time or not total_rows:
            return 0.0
            
        # Target: process 1000 rows per second
        target_rate = 1000
        actual_rate = total_rows / total_time
        
        return min(1.0, actual_rate / target_rate)
    
    def _calculate_cost_efficiency(self) -> float:
        """Calculate cost efficiency score (simplified)"""
        # Based on compute resources and processing time
        processing_hours = self.monitor.metrics.total_processing_time / 3600
        records_processed = self.monitor.metrics.total_rows_processed
        
        if not processing_hours or not records_processed:
            return 0.0
            
        # Target: 1M records per hour
        target_rate = 1000000
        actual_rate = records_processed / processing_hours
        
        return min(1.0, actual_rate / target_rate)
    
    def _log_benchmark_results(self, benchmark_results: Dict):
        """Log benchmark results to console and MLflow"""
        self.logger.info("🏃 Load Testing Results:")
        self.logger.info("-" * 50)
        
        for scenario, results in benchmark_results.items():
            throughput = results['throughput']
            processing_time = results['processing_time']
            
            self.logger.info(f"{scenario}: {throughput:.1f} records/sec ({processing_time:.2f}s)")
        
        # Log scaling efficiency
        if len(benchmark_results) >= 2:
            scenarios = sorted(benchmark_results.keys(), key=lambda x: int(x.replace('x', '')))
            base_throughput = benchmark_results[scenarios[0]]['throughput']
            
            self.logger.info("\n📈 Scaling Analysis:")
            for scenario in scenarios[1:]:
                multiplier = int(scenario.replace('x', ''))
                actual_throughput = benchmark_results[scenario]['throughput']
                expected_throughput = base_throughput * multiplier
                efficiency = (actual_throughput / expected_throughput) * 100
                
                self.logger.info(f"{scenario}: {efficiency:.1f}% scaling efficiency")
    
    def _generate_training_report(self, scalability_metrics):
        """Generate comprehensive training report"""
        report_path = f"results/training/training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        # Generate monitoring report
        monitoring_report = self.monitor.generate_monitoring_report(report_path.replace('.txt', '_monitoring.txt'))
        
        # Generate training-specific report
        training_report = self._create_training_report()
        
        # Combine reports
        combined_report = f"""
{training_report}

{monitoring_report}
"""
        
        with open(report_path, 'w') as f:
            f.write(combined_report)
        
        self.logger.info(f"📄 Training report saved to: {report_path}")
        
        # Export metrics to JSON
        metrics_path = report_path.replace('.txt', '_metrics.json')
        if self.monitor:
            self.monitor.export_metrics_json(metrics_path)
    
    def _create_training_report(self) -> str:
        """Create training-specific report"""
        report_lines = [
            "=" * 80,
            "🚂 TRAINING PIPELINE REPORT",
            "=" * 80,
            f"Training Start: {self.training_metrics['start_time']}",
            f"Training End: {self.training_metrics['end_time']}",
            f"Total Training Time: {self.training_metrics['total_training_time']:.2f} seconds",
            "",
            "📊 TRAINING RESULTS:",
            "-" * 40,
            f"Stocks Processed: {len(self.training_metrics['stocks_processed'])}",
            f"Models Trained: {len(self.training_metrics['models_trained'])}",
        ]
        
        # Add individual stock results
        for model_info in self.training_metrics.get('models_trained', []):
            stock = model_info['stock']
            training_time = model_info['training_time']
            feature_count = model_info['feature_count']
            
            report_lines.extend([
                f"",
                f"📈 {stock}:",
                f"  Training Time: {training_time:.2f}s",
                f"  Features: {feature_count}",
            ])
            
            # Add model metrics if available
            model_metrics = model_info.get('model_metrics', {})
            for metric, value in model_metrics.items():
                if isinstance(value, (int, float)):
                    report_lines.append(f"  {metric}: {value:.4f}")
        
        # Add training efficiency if available
        if 'training_efficiency' in self.training_metrics:
            report_lines.extend([
                "",
                "⚡ TRAINING EFFICIENCY:",
                "-" * 40,
            ])
            
            for metric, score in self.training_metrics['training_efficiency'].items():
                report_lines.append(f"{metric}: {score:.3f}")
        
        return "\n".join(report_lines)


def create_spark_session(config: Config) -> SparkSession:
    """Create Spark session for training with monitoring"""
    
    builder = SparkSession.builder \
        .appName(f"{config.spark.app_name}_Training_With_Monitoring") \
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
    parser = argparse.ArgumentParser(description='Train models with scalability monitoring')
    
    parser.add_argument(
        '--stocks', 
        type=str, 
        help='Comma-separated list of stock symbols to train'
    )
    parser.add_argument(
        '--all', 
        action='store_true', 
        help='Train models for all available stocks'
    )
    parser.add_argument(
        '--no-monitoring', 
        action='store_true', 
        help='Disable scalability monitoring'
    )
    parser.add_argument(
        '--load-test', 
        action='store_true', 
        help='Run load testing benchmark'
    )
    parser.add_argument(
        '--monitoring-config', 
        type=str, 
        help='Path to custom monitoring configuration JSON file'
    )
    
    return parser.parse_args()


def main():
    """Main training pipeline with monitoring"""
    
    print("🚂 Starting Monitored Model Training Pipeline")
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
        
        # Create monitored trainer
        enable_monitoring = config.monitoring.enabled and not args.no_monitoring
        trainer = MonitoredModelTrainer(config, spark, enable_monitoring)
        
        # Determine stock selection
        stock_selection = None
        if args.stocks:
            stock_selection = [s.strip().upper() for s in args.stocks.split(',')]
            print(f"🎯 Training specific stocks: {stock_selection}")
        elif args.all:
            stock_selection = config.data.available_stocks
            print(f"🎯 Training all available stocks: {stock_selection}")
        else:
            stock_selection = config.data.training_stocks
            print(f"🎯 Training configured stocks: {stock_selection}")
        
        # Validate stock symbols
        invalid_stocks = [s for s in stock_selection if s not in config.data.available_stocks]
        if invalid_stocks:
            print(f"❌ Invalid stock symbols: {invalid_stocks}")
            print(f"Available stocks: {config.data.available_stocks}")
            return
        
        # Run training with monitoring
        results = trainer.train_models(stock_selection, args.load_test)
        
        # Print summary
        print("\n🎉 Training pipeline completed successfully!")
        print(f"⏱️  Total time: {results['total_training_time']:.2f} seconds")
        print(f"📊 Stocks processed: {len(results['stocks_processed'])}")
        print(f"🤖 Models trained: {len(results['models_trained'])}")
        
        if enable_monitoring and 'scalability_metrics' in results:
            metrics = results['scalability_metrics']
            print(f"📈 Throughput: {metrics.throughput_records_per_second:.1f} records/sec")
            print(f"💾 Peak memory: {metrics.peak_memory_usage_mb:.1f} MB")
            print(f"🎯 Scalability score: {metrics.linear_scalability_score:.3f}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Training pipeline failed: {str(e)}")
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