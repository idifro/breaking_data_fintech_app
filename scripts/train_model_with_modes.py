#!/usr/bin/env python3
"""
Multi-Mode Training Script for Data Engineering at Scale Project
Supports three training approaches: Spark GBT, Spark RF, and Hybrid Sklearn

CONDA ENVIRONMENT: breaking_data
Usage: 
    conda activate breaking_data
    
    # Train with specific mode
    python scripts/train_model_with_modes.py --mode spark_gbt
    python scripts/train_model_with_modes.py --mode spark_rf
    python scripts/train_model_with_modes.py --mode hybrid_sklearn
    
    # Run scalability evaluation across all modes
    python scripts/train_model_with_modes.py --scalability-evaluation
    
    # Train with specific data volume
    python scripts/train_model_with_modes.py --mode spark_gbt --data-volume small
    python scripts/train_model_with_modes.py --mode spark_rf --data-volume medium
    
    # Run all modes for comparison
    python scripts/train_model_with_modes.py --run-all-modes
"""

import sys
import os
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import json
import yaml

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
from src.scalability_monitor import ScalabilityMonitor
from src.monitoring_config_manager import get_monitoring_config_manager


class MultiModeConfig:
    """Configuration manager for multi-mode training"""
    
    def __init__(self, config_path: str = "config/config_modes.yaml"):
        """Load multi-mode configuration"""
        self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Multi-mode config file not found: {config_path}")
        
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        # Also load base config for data and feature settings
        self.base_config = Config()
        
        self.logger = Logger().get_logger()
        self.logger.info(f"Loaded multi-mode configuration from: {config_path}")
    
    @property
    def training_modes(self) -> Dict:
        return self._config['training_modes']
    
    @property
    def scalability_testing(self) -> Dict:
        return self._config['scalability_testing']
    
    @property
    def evaluation(self) -> Dict:
        return self._config['evaluation']
    
    @property
    def monitoring(self) -> Dict:
        return self._config['monitoring']
    
    @property
    def spark_config(self) -> Dict:
        return self._config['spark']
    
    def get_mode_config(self, mode: str) -> Dict:
        """Get configuration for specific training mode"""
        mode_key = f"{mode}_mode"
        if mode_key not in self._config:
            raise ValueError(f"Unknown training mode: {mode}")
        return self._config[mode_key]
    
    def get_stock_selection(self, data_volume: str) -> List[str]:
        """Get stock selection for specific data volume"""
        if data_volume == "full":
            return self.base_config.data.available_stocks
        
        selections = self.scalability_testing['stock_selections']
        if data_volume not in selections:
            raise ValueError(f"Unknown data volume: {data_volume}")
        
        return selections[data_volume]
    
    def get_data_volumes(self) -> List[str]:
        """Get available data volume sizes"""
        return list(self.scalability_testing['data_volumes'].keys())


class MultiModeTrainer:
    """Multi-mode training coordinator with scalability evaluation"""
    
    def __init__(self, multi_config: MultiModeConfig, spark: SparkSession):
        self.multi_config = multi_config
        self.spark = spark
        self.logger = Logger().get_logger()
        
        # Initialize data processor with base config
        self.data_processor = DataProcessor(multi_config.base_config, spark)
        
        # Initialize monitoring
        self.enable_monitoring = multi_config.monitoring['enabled']
        if self.enable_monitoring:
            monitoring_config = {
                'enabled': True,
                'detailed_metrics': multi_config.monitoring['detailed_metrics'],
                'resource_monitoring_interval': multi_config.monitoring['resource_monitoring_interval'],
                'experiment_name': 'multi_mode_scalability_monitoring'
            }
            self.monitor = ScalabilityMonitor(multi_config.base_config, spark, monitoring_config)
        else:
            self.monitor = None
        
        # Training results storage
        self.training_results = {}
        
        self.logger.info("🚀 Multi-mode trainer initialized")
    
    def train_with_mode(self, mode: str, data_volume: str = "full", 
                       enable_scalability_eval: bool = False) -> Dict:
        """Train model with specific mode and data volume"""
        
        self.logger.info(f"🎯 Starting training - Mode: {mode}, Data Volume: {data_volume}")
        
        # Get mode configuration
        mode_config = self.multi_config.get_mode_config(mode)
        
        # Get stock selection for data volume
        stock_selection = self.multi_config.get_stock_selection(data_volume)
        
        self.logger.info(f"📊 Training with {len(stock_selection)} stocks: {stock_selection}")
        
        # Start monitoring if enabled
        if self.monitor:
            run_name = f"{mode}_{data_volume}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.monitor.start_monitoring(run_name)
        
        # Route to appropriate trainer based on mode
        if mode == "spark_gbt":
            result = self._train_spark_gbt(mode_config, stock_selection, enable_scalability_eval)
        elif mode == "spark_rf":
            result = self._train_spark_rf(mode_config, stock_selection, enable_scalability_eval)
        elif mode == "hybrid_sklearn":
            result = self._train_hybrid_sklearn(mode_config, stock_selection, enable_scalability_eval)
        else:
            raise ValueError(f"Unknown training mode: {mode}")
        
        # Stop monitoring and collect metrics
        if self.monitor:
            scalability_metrics = self.monitor.stop_monitoring()
            result['scalability_metrics'] = scalability_metrics
        
        # Store results
        self.training_results[f"{mode}_{data_volume}"] = result
        
        self.logger.info(f"✅ Training completed - Mode: {mode}, Data Volume: {data_volume}")
        return result
    
    def _train_spark_gbt(self, mode_config: Dict, stock_selection: List[str], 
                        enable_scalability_eval: bool) -> Dict:
        """Train using Spark ML GBT (enhanced current implementation)"""
        from src.model_training_modes.spark_gbt_trainer import SparkGBTTrainer
        
        self.logger.info("🌳 Training with Spark ML GBT (enhanced current implementation)")
        
        trainer = SparkGBTTrainer(mode_config, self.multi_config.base_config, self.spark, monitor=self.monitor)
        result = trainer.train_unified_model(stock_selection, enable_scalability_eval)
        
        return result
    
    def _train_spark_rf(self, mode_config: Dict, stock_selection: List[str], 
                       enable_scalability_eval: bool) -> Dict:
        """Train using Spark ML Random Forest (distributed training)"""
        from src.model_training_modes.spark_rf_trainer import SparkRFTrainer
        
        self.logger.info("🌲 Training with Spark ML Random Forest (distributed)")
        
        trainer = SparkRFTrainer(mode_config, self.multi_config.base_config, self.spark, monitor=self.monitor)
        result = trainer.train_unified_model(stock_selection, enable_scalability_eval)
        
        return result
    
    def _train_hybrid_sklearn(self, mode_config: Dict, stock_selection: List[str], 
                             enable_scalability_eval: bool) -> Dict:
        """Train using Hybrid Spark + Scikit-Learn approach"""
        from src.model_training_modes.hybrid_sklearn_trainer import HybridSklearnTrainer
        
        self.logger.info("🔄 Training with Hybrid Spark + Scikit-Learn approach")
        
        trainer = HybridSklearnTrainer(mode_config, self.multi_config.base_config, self.spark, monitor=self.monitor)
        result = trainer.train_unified_model(stock_selection, enable_scalability_eval)
        
        return result
    
    def run_scalability_evaluation(self) -> Dict:
        """Run comprehensive scalability evaluation across all modes and data volumes"""
        
        self.logger.info("📈 Starting comprehensive scalability evaluation")
        
        evaluation_results = {
            'modes': {},
            'data_volumes': self.multi_config.get_data_volumes(),
            'comparison_metrics': {},
            'scalability_analysis': {}
        }
        
        # Get available modes
        available_modes = self.multi_config.training_modes['available_modes']
        data_volumes = self.multi_config.get_data_volumes()
        
        start_time = time.time()
        
        # Run training for each mode and data volume combination
        for mode in available_modes:
            self.logger.info(f"🔍 Evaluating mode: {mode}")
            
            mode_results = {}
            for volume in data_volumes:
                self.logger.info(f"  📊 Data volume: {volume}")
                
                try:
                    result = self.train_with_mode(mode, volume, enable_scalability_eval=True)
                    mode_results[volume] = result
                    
                    # Log progress
                    if 'scalability_metrics' in result:
                        metrics = result['scalability_metrics']
                        self.logger.info(f"    ⚡ Throughput: {metrics.throughput_records_per_second:.1f} records/sec")
                        self.logger.info(f"    💾 Memory: {metrics.peak_memory_usage_mb:.1f} MB")
                        
                except Exception as e:
                    self.logger.error(f"    ❌ Failed: {str(e)}")
                    mode_results[volume] = {'error': str(e)}
            
            evaluation_results['modes'][mode] = mode_results
        
        # Calculate comparison metrics
        evaluation_results['comparison_metrics'] = self._calculate_comparison_metrics(evaluation_results['modes'])
        
        # Perform scalability analysis
        evaluation_results['scalability_analysis'] = self._analyze_scalability_trends(evaluation_results['modes'])
        
        # Generate comprehensive report
        total_time = time.time() - start_time
        evaluation_results['total_evaluation_time'] = total_time
        
        self._generate_scalability_report(evaluation_results)
        
        self.logger.info(f"✅ Scalability evaluation completed in {total_time:.2f} seconds")
        
        return evaluation_results
    
    def _calculate_comparison_metrics(self, modes_results: Dict) -> Dict:
        """Calculate comparison metrics across modes and data volumes"""
        
        comparison = {
            'training_time_comparison': {},
            'memory_efficiency_comparison': {},
            'accuracy_comparison': {},
            'scalability_scores': {}
        }
        
        # Compare training times
        for mode, volumes in modes_results.items():
            training_times = {}
            for volume, result in volumes.items():
                if 'error' not in result and 'training_time' in result:
                    training_times[volume] = result['training_time']
            comparison['training_time_comparison'][mode] = training_times
        
        # Compare memory efficiency
        for mode, volumes in modes_results.items():
            memory_usage = {}
            for volume, result in volumes.items():
                if 'error' not in result and 'scalability_metrics' in result:
                    memory_usage[volume] = result['scalability_metrics'].peak_memory_usage_mb
            comparison['memory_efficiency_comparison'][mode] = memory_usage
        
        # Compare model accuracy
        for mode, volumes in modes_results.items():
            accuracy_metrics = {}
            for volume, result in volumes.items():
                if 'error' not in result and 'metrics' in result:
                    accuracy_metrics[volume] = result['metrics']['test_metrics']
            comparison['accuracy_comparison'][mode] = accuracy_metrics
        
        # Calculate scalability scores
        for mode, volumes in modes_results.items():
            scalability_scores = {}
            for volume, result in volumes.items():
                if 'error' not in result and 'scalability_metrics' in result:
                    scalability_scores[volume] = result['scalability_metrics'].linear_scalability_score
            comparison['scalability_scores'][mode] = scalability_scores
        
        return comparison
    
    def _analyze_scalability_trends(self, modes_results: Dict) -> Dict:
        """Analyze scalability trends and patterns"""
        
        analysis = {
            'linear_scalability_assessment': {},
            'memory_growth_patterns': {},
            'performance_degradation_points': {},
            'mode_rankings': {}
        }
        
        # Analyze linear scalability for each mode
        data_volumes = self.multi_config.get_data_volumes()
        volume_sizes = [self.multi_config.scalability_testing['data_volumes'][vol] for vol in data_volumes]
        
        for mode, volumes in modes_results.items():
            # Extract training times for scalability assessment
            times = []
            volumes_list = []
            
            for vol in data_volumes:
                if vol in volumes and 'error' not in volumes[vol]:
                    if 'training_time' in volumes[vol]:
                        times.append(volumes[vol]['training_time'])
                        volumes_list.append(self.multi_config.scalability_testing['data_volumes'][vol])
            
            if len(times) >= 2:
                # Calculate linear scalability score
                # Ideal: time should scale linearly with data volume
                scalability_score = self._calculate_linear_scalability_score(volumes_list, times)
                analysis['linear_scalability_assessment'][mode] = {
                    'score': scalability_score,
                    'data_points': list(zip(volumes_list, times))
                }
            
            # Analyze memory growth patterns
            memory_usage = []
            for vol in data_volumes:
                if vol in volumes and 'error' not in volumes[vol]:
                    if 'scalability_metrics' in volumes[vol]:
                        memory_usage.append(volumes[vol]['scalability_metrics'].peak_memory_usage_mb)
            
            if len(memory_usage) >= 2:
                memory_growth_rate = (memory_usage[-1] - memory_usage[0]) / (len(memory_usage) - 1)
                analysis['memory_growth_patterns'][mode] = {
                    'growth_rate_mb_per_volume': memory_growth_rate,
                    'data_points': list(zip(data_volumes[:len(memory_usage)], memory_usage))
                }
        
        # Rank modes by overall performance
        mode_scores = {}
        for mode, volumes in modes_results.items():
            # Calculate overall score based on training time, memory efficiency, and accuracy
            scores = []
            for vol, result in volumes.items():
                if 'error' not in result:
                    # Normalize and combine metrics (lower is better for time/memory, higher for accuracy)
                    time_score = 1.0 / (result.get('training_time', float('inf')) + 1e-6)
                    memory_score = 1.0 / (result.get('scalability_metrics', type('', (), {'peak_memory_usage_mb': float('inf')})).peak_memory_usage_mb + 1e-6)
                    
                    if 'metrics' in result and 'test_metrics' in result['metrics']:
                        accuracy_score = result['metrics']['test_metrics'].get('r2', 0)
                    else:
                        accuracy_score = 0
                    
                    # Weighted combination
                    combined_score = 0.4 * time_score + 0.3 * memory_score + 0.3 * accuracy_score
                    scores.append(combined_score)
            
            if scores:
                mode_scores[mode] = sum(scores) / len(scores)
        
        # Sort modes by performance
        ranked_modes = sorted(mode_scores.items(), key=lambda x: x[1], reverse=True)
        analysis['mode_rankings'] = {
            'ranking': [mode for mode, score in ranked_modes],
            'scores': dict(ranked_modes)
        }
        
        return analysis
    
    def _calculate_linear_scalability_score(self, data_volumes: List[int], training_times: List[float]) -> float:
        """Calculate linear scalability score (1.0 = perfect linear scaling, 0.0 = no scaling)"""
        if len(data_volumes) < 2 or len(training_times) < 2:
            return 0.0
        
        # Calculate correlation coefficient between data volume and training time
        import numpy as np
        
        # Normalize to relative scales
        vol_ratios = [vol / data_volumes[0] for vol in data_volumes]
        time_ratios = [time / training_times[0] for time in training_times]
        
        # Perfect linear scaling would have time_ratio == vol_ratio
        # Calculate deviation from perfect linear scaling
        deviations = [abs(time_ratio - vol_ratio) for time_ratio, vol_ratio in zip(time_ratios, vol_ratios)]
        avg_deviation = sum(deviations) / len(deviations)
        
        # Convert to score (lower deviation = higher score)
        linear_scalability_score = max(0.0, 1.0 - avg_deviation)
        
        return linear_scalability_score
    
    def _generate_scalability_report(self, evaluation_results: Dict):
        """Generate comprehensive scalability evaluation report"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f"results/scalability/scalability_evaluation_report_{timestamp}.txt"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        report_lines = [
            "=" * 80,
            "🚀 DATA ENGINEERING AT SCALE - SCALABILITY EVALUATION REPORT",
            "=" * 80,
            f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Evaluation Time: {evaluation_results['total_evaluation_time']:.2f} seconds",
            "",
            "📊 TRAINING MODES EVALUATED:",
            "-" * 40,
        ]
        
        for mode in evaluation_results['modes'].keys():
            mode_desc = self.multi_config.training_modes['descriptions'][mode]
            report_lines.append(f"• {mode}: {mode_desc}")
        
        report_lines.extend([
            "",
            "📈 DATA VOLUME SCALABILITY:",
            "-" * 40,
        ])
        
        for volume in evaluation_results['data_volumes']:
            volume_size = self.multi_config.scalability_testing['data_volumes'][volume]
            report_lines.append(f"• {volume}: {volume_size} stocks")
        
        # Add performance comparison
        comparison = evaluation_results['comparison_metrics']
        
        report_lines.extend([
            "",
            "⚡ TRAINING TIME COMPARISON:",
            "-" * 40,
        ])
        
        for mode, times in comparison['training_time_comparison'].items():
            report_lines.append(f"\n{mode.upper()}:")
            for volume, time_val in times.items():
                report_lines.append(f"  {volume}: {time_val:.2f}s")
        
        report_lines.extend([
            "",
            "💾 MEMORY USAGE COMPARISON:",
            "-" * 40,
        ])
        
        for mode, memory in comparison['memory_efficiency_comparison'].items():
            report_lines.append(f"\n{mode.upper()}:")
            for volume, mem_val in memory.items():
                report_lines.append(f"  {volume}: {mem_val:.1f} MB")
        
        # Add scalability analysis
        analysis = evaluation_results['scalability_analysis']
        
        report_lines.extend([
            "",
            "🎯 SCALABILITY ANALYSIS:",
            "-" * 40,
        ])
        
        if 'mode_rankings' in analysis:
            rankings = analysis['mode_rankings']
            report_lines.append("\nMODE PERFORMANCE RANKING:")
            for i, mode in enumerate(rankings['ranking'], 1):
                score = rankings['scores'][mode]
                report_lines.append(f"  {i}. {mode}: {score:.4f}")
        
        if 'linear_scalability_assessment' in analysis:
            report_lines.append("\nLINEAR SCALABILITY SCORES:")
            for mode, assessment in analysis['linear_scalability_assessment'].items():
                score = assessment['score']
                report_lines.append(f"  {mode}: {score:.3f}")
        
        # Add recommendations
        report_lines.extend([
            "",
            "🎯 RECOMMENDATIONS:",
            "-" * 40,
        ])
        
        if 'mode_rankings' in analysis and analysis['mode_rankings']['ranking']:
            best_mode = analysis['mode_rankings']['ranking'][0]
            report_lines.append(f"• Best Overall Performance: {best_mode}")
            report_lines.append(f"• Recommended for Production: {best_mode}")
        
        # Write report
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        self.logger.info(f"📄 Scalability evaluation report saved: {report_path}")
        
        # Also save JSON version for analysis
        json_path = report_path.replace('.txt', '.json')
        with open(json_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        self.logger.info(f"💾 Detailed evaluation data saved: {json_path}")


def create_spark_session(multi_config: MultiModeConfig) -> SparkSession:
    """Create Spark session optimized for multi-mode training"""
    
    spark_config = multi_config.spark_config
    
    builder = SparkSession.builder \
        .appName(spark_config['app_name']) \
        .master(spark_config['resources']['master']) \
        .config("spark.driver.memory", spark_config['resources']['driver_memory']) \
        .config("spark.executor.memory", spark_config['resources']['executor_memory']) \
        .config("spark.executor.cores", str(spark_config['resources']['executor_cores'])) \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    
    # Add additional configurations
    for key, value in spark_config['configs'].items():
        builder = builder.config(key, value)
    
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    
    return spark


def parse_arguments():
    """Parse command line arguments"""
    
    parser = argparse.ArgumentParser(description='Multi-mode training with scalability evaluation')
    
    # Mode selection
    parser.add_argument(
        '--mode', 
        type=str,
        choices=['spark_gbt', 'spark_rf', 'hybrid_sklearn'],
        help='Training mode to use'
    )
    
    # Data volume selection
    parser.add_argument(
        '--data-volume',
        type=str,
        choices=['small', 'medium', 'large', 'full'],
        default='full',
        help='Data volume size for training'
    )
    
    # Scalability evaluation
    parser.add_argument(
        '--scalability-evaluation',
        action='store_true',
        help='Run comprehensive scalability evaluation across all modes and data volumes'
    )
    
    # Run all modes
    parser.add_argument(
        '--run-all-modes',
        action='store_true',
        help='Run training with all available modes'
    )
    
    # Configuration file
    parser.add_argument(
        '--config',
        type=str,
        default='config/config_modes.yaml',
        help='Path to multi-mode configuration file'
    )
    
    # Monitoring options
    parser.add_argument(
        '--no-monitoring',
        action='store_true',
        help='Disable monitoring'
    )
    
    return parser.parse_args()


def main():
    """Main multi-mode training pipeline"""
    
    print("🚀 Starting Multi-Mode Training Pipeline for Data Engineering at Scale")
    print("=" * 80)
    
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Load environment and configuration
        load_environment()
        multi_config = MultiModeConfig(args.config)
        
        print(f"🔧 Loaded configuration from: {args.config}")
        print(f"📊 Available modes: {multi_config.training_modes['available_modes']}")
        print(f"📈 Data volumes: {multi_config.get_data_volumes()}")
        
        # Override monitoring setting
        if args.no_monitoring:
            multi_config._config['monitoring']['enabled'] = False
            print("⚠️  Monitoring disabled")
        
        # Create Spark session
        spark = create_spark_session(multi_config)
        print("🔧 Spark session created successfully")
        
        # Create multi-mode trainer
        trainer = MultiModeTrainer(multi_config, spark)
        
        results = {}
        
        # Execute based on arguments
        if args.scalability_evaluation:
            print("📈 Running comprehensive scalability evaluation...")
            results = trainer.run_scalability_evaluation()
            
        elif args.run_all_modes:
            print("🔄 Running all training modes...")
            for mode in multi_config.training_modes['available_modes']:
                print(f"\n🎯 Training with mode: {mode}")
                result = trainer.train_with_mode(mode, args.data_volume)
                results[f"{mode}_{args.data_volume}"] = result
                
        elif args.mode:
            print(f"🎯 Training with mode: {args.mode}, data volume: {args.data_volume}")
            results = trainer.train_with_mode(args.mode, args.data_volume)
            
        else:
            print("❌ Please specify --mode, --run-all-modes, or --scalability-evaluation")
            return
        
        # Print summary
        print("\n" + "=" * 80)
        print("🎉 Multi-mode training completed successfully!")
        
        if isinstance(results, dict):
            if 'total_evaluation_time' in results:
                print(f"⏱️  Total evaluation time: {results['total_evaluation_time']:.2f} seconds")
            elif len(results) == 1:
                # Single mode result
                result = list(results.values())[0]
                if 'training_time' in result:
                    print(f"⏱️  Training time: {result['training_time']:.2f} seconds")
                if 'scalability_metrics' in result:
                    metrics = result['scalability_metrics']
                    print(f"📈 Throughput: {metrics.throughput_records_per_second:.1f} records/sec")
                    print(f"💾 Peak memory: {metrics.peak_memory_usage_mb:.1f} MB")
            else:
                print(f"📊 Completed {len(results)} training runs")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Multi-mode training failed: {str(e)}")
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