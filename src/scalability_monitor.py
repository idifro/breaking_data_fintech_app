#!/usr/bin/env python3
"""
Scalability Monitoring Module for ML Pipeline
Provides comprehensive monitoring for data volume, performance, and resource utilization scalability

This module is designed for Data Engineering at Scale projects to demonstrate
pipeline scalability metrics and identify performance bottlenecks.
"""

import time
import psutil
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import logging
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import gc

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, count, sum as spark_sum, avg, max as spark_max
from pyspark.sql.types import *
import mlflow
from mlflow.tracking import MlflowClient

from src.utils import Config, Logger


@dataclass
class ScalabilityMetrics:
    """Data class for storing scalability metrics"""
    
    # Data Volume Metrics
    total_rows_processed: int = 0
    data_size_mb: float = 0.0
    avg_partition_size_mb: float = 0.0
    partition_count: int = 0
    partition_efficiency_score: float = 0.0
    
    # Performance Metrics
    total_processing_time: float = 0.0
    avg_processing_time_per_record: float = 0.0
    throughput_records_per_second: float = 0.0
    feature_engineering_time: float = 0.0
    model_inference_time: float = 0.0
    
    # Resource Utilization Metrics
    peak_memory_usage_mb: float = 0.0
    avg_memory_usage_mb: float = 0.0
    peak_cpu_usage_percent: float = 0.0
    avg_cpu_usage_percent: float = 0.0
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    
    # Scalability Indicators
    linear_scalability_score: float = 0.0
    resource_efficiency_score: float = 0.0
    cost_efficiency_score: float = 0.0
    bottleneck_indicators: Dict = None
    
    # Timestamps
    start_time: datetime = None
    end_time: datetime = None
    
    def __post_init__(self):
        if self.bottleneck_indicators is None:
            self.bottleneck_indicators = {}


class ScalabilityMonitor:
    """
    Comprehensive scalability monitoring for ML pipelines
    
    Monitors:
    - Data volume scalability (rows, partitions, storage)
    - Processing performance (throughput, latency) 
    - Resource utilization (CPU, memory, I/O)
    - Cost efficiency metrics
    - Bottleneck identification
    """
    
    def __init__(self, config: Config, spark: SparkSession, monitoring_config: Dict = None):
        """
        Initialize scalability monitor
        
        Args:
            config: Pipeline configuration
            spark: Spark session
            monitoring_config: Optional monitoring-specific configuration
        """
        self.config = config
        self.spark = spark
        self.logger = Logger().get_logger()
        
        # Monitoring configuration with defaults
        self.monitoring_config = monitoring_config or {}
        self.enabled = self.monitoring_config.get('enabled', True)
        self.detailed_metrics = self.monitoring_config.get('detailed_metrics', True)
        self.resource_monitoring_interval = self.monitoring_config.get('resource_monitoring_interval', 1.0)
        
        # Metrics storage
        self.metrics = ScalabilityMetrics()
        self.resource_samples = []
        self.performance_samples = []
        
        # Monitoring state
        self.monitoring_active = False
        self.start_time = None
        
        # MLflow setup for monitoring
        self.monitoring_experiment_name = self.monitoring_config.get(
            'experiment_name', 'pipeline_scalability_monitoring'
        )
        
        # Check Spark API compatibility
        self._check_spark_compatibility()
        self._setup_monitoring_experiment()
    
    def _check_spark_compatibility(self):
        """Check Spark API compatibility and log warnings for known issues"""
        try:
            status_tracker = self.spark.sparkContext.statusTracker()
            
            # Test for deprecated getExecutorInfos method
            if not hasattr(status_tracker, 'getExecutorInfos'):
                self.logger.info(
                    "Spark StatusTracker.getExecutorInfos() not available - using fallback executor estimation"
                )
            
            # Log Spark version for debugging
            spark_version = self.spark.version
            self.logger.info(f"Monitoring compatible with Spark version: {spark_version}")
            
        except Exception as e:
            self.logger.warning(f"Spark compatibility check failed: {e}")
    
    def _setup_monitoring_experiment(self):
        """Setup dedicated MLflow experiment for monitoring"""
        try:
            mlflow.set_tracking_uri(str(self.config.paths.mlflow_tracking_dir))
            
            # Create or get monitoring experiment
            client = MlflowClient()
            try:
                experiment = client.get_experiment_by_name(self.monitoring_experiment_name)
                if experiment is None:
                    experiment_id = client.create_experiment(self.monitoring_experiment_name)
                    self.logger.info(f"Created monitoring experiment: {self.monitoring_experiment_name}")
                else:
                    experiment_id = experiment.experiment_id
                    self.logger.info(f"Using existing monitoring experiment: {self.monitoring_experiment_name}")
                    
                mlflow.set_experiment(self.monitoring_experiment_name)
                
            except Exception as e:
                self.logger.warning(f"Could not setup monitoring experiment: {e}")
                
        except Exception as e:
            self.logger.warning(f"MLflow setup failed for monitoring: {e}")
    
    def start_monitoring(self, operation_type: str = "general") -> None:
        """
        Start comprehensive monitoring session
        
        Args:
            operation_type: Type of operation being monitored (training, inference, etc.)
        """
        if not self.enabled:
            return
            
        self.logger.info(f"🔍 Starting scalability monitoring for {operation_type}")
        
        self.monitoring_active = True
        self.start_time = datetime.now()
        self.metrics.start_time = self.start_time
        
        # Reset metrics
        self.resource_samples = []
        self.performance_samples = []
        
        # Start resource monitoring in background
        if self.detailed_metrics:
            self._start_resource_monitoring()
    
    def stop_monitoring(self) -> ScalabilityMetrics:
        """
        Stop monitoring and calculate final metrics
        
        Returns:
            ScalabilityMetrics: Comprehensive scalability metrics
        """
        if not self.enabled or not self.monitoring_active:
            return self.metrics
            
        self.monitoring_active = False
        end_time = datetime.now()
        self.metrics.end_time = end_time
        self.metrics.total_processing_time = (end_time - self.start_time).total_seconds()
        
        # Calculate final metrics
        self._calculate_performance_metrics()
        self._calculate_resource_metrics()
        self._calculate_scalability_scores()
        self._identify_bottlenecks()
        
        # Log to MLflow
        self._log_metrics_to_mlflow()
        
        self.logger.info(f"✅ Monitoring completed. Total time: {self.metrics.total_processing_time:.2f}s")
        
        return self.metrics
    
    def monitor_data_volume(self, operation_name: str = "data_analysis") -> Dict:
        """
        Monitor data volume scalability metrics
        
        Args:
            operation_name: Name of the data operation being monitored
            
        Returns:
            Dict: Data volume metrics
        """
        if not self.enabled:
            return {}
            
        self.logger.info(f"📊 Monitoring data volume for {operation_name}")
        
        try:
            data_metrics = {}
            total_rows = 0
            total_size_mb = 0
            partition_stats = []
            
            # Analyze each stock table
            for stock in self.config.data.available_stocks:
                try:
                    table_path = f"delta_tables/stock_{stock}"
                    
                    # Check if table exists
                    if not Path(table_path).exists():
                        continue
                        
                    df = self.spark.read.format("delta").load(table_path)
                    
                    # Get basic stats
                    row_count = df.count()
                    partition_count = df.rdd.getNumPartitions()
                    
                    # Estimate data size
                    estimated_size = self._estimate_dataframe_size(df, sample_size=1000)
                    
                    total_rows += row_count
                    total_size_mb += estimated_size
                    
                    partition_stats.append({
                        'stock': stock,
                        'rows': row_count,
                        'partitions': partition_count,
                        'rows_per_partition': row_count / max(partition_count, 1),
                        'estimated_size_mb': estimated_size
                    })
                    
                    self.logger.debug(f"  {stock}: {row_count:,} rows, {partition_count} partitions, {estimated_size:.1f} MB")
                    
                except Exception as e:
                    self.logger.warning(f"Could not analyze {stock}: {e}")
                    
            # Update metrics
            self.metrics.total_rows_processed = total_rows
            self.metrics.data_size_mb = total_size_mb
            self.metrics.partition_count = sum(p['partitions'] for p in partition_stats)
            
            if partition_stats:
                self.metrics.avg_partition_size_mb = np.mean([p['estimated_size_mb'] / p['partitions'] for p in partition_stats])
                self.metrics.partition_efficiency_score = self._calculate_partition_efficiency(partition_stats)
            
            data_metrics = {
                'total_rows': total_rows,
                'total_size_mb': total_size_mb,
                'partition_efficiency': self.metrics.partition_efficiency_score,
                'per_stock_stats': partition_stats
            }
            
            self.logger.info(f"📊 Data volume: {total_rows:,} rows, {total_size_mb:.1f} MB, {len(partition_stats)} tables")
            
            return data_metrics
            
        except Exception as e:
            self.logger.error(f"Data volume monitoring failed: {e}")
            return {}
    
    def monitor_performance(self, operation_name: str, start_time: float, records_processed: int = 0) -> Dict:
        """
        Monitor performance metrics for an operation
        
        Args:
            operation_name: Name of the operation
            start_time: Operation start time (from time.time())
            records_processed: Number of records processed
            
        Returns:
            Dict: Performance metrics
        """
        if not self.enabled:
            return {}
            
        end_time = time.time()
        duration = end_time - start_time
        
        performance_metrics = {
            'operation': operation_name,
            'duration_seconds': duration,
            'records_processed': records_processed,
            'throughput_records_per_second': records_processed / max(duration, 0.001),
            'avg_time_per_record_ms': (duration * 1000) / max(records_processed, 1)
        }
        
        # Store sample
        self.performance_samples.append(performance_metrics)
        
        # Update cumulative metrics
        if operation_name == 'feature_engineering':
            self.metrics.feature_engineering_time += duration
        elif operation_name == 'model_inference':
            self.metrics.model_inference_time += duration
            
        self.logger.info(
            f"⚡ {operation_name}: {duration:.2f}s, "
            f"{performance_metrics['throughput_records_per_second']:.1f} records/sec"
        )
        
        return performance_metrics
    
    def monitor_resource_usage(self) -> Dict:
        """
        Monitor current resource usage
        
        Returns:
            Dict: Resource usage metrics
        """
        if not self.enabled:
            return {}
            
        try:
            # System resources
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=None)
            disk_io = psutil.disk_io_counters()
            
            # Spark resources
            spark_context = self.spark.sparkContext
            status_tracker = spark_context.statusTracker()
            
            # Safely get executor count with multiple fallbacks for API compatibility
            spark_executors = 1  # Default fallback
            try:
                # Method 1: Try the old getExecutorInfos method (deprecated but may exist)
                if hasattr(status_tracker, 'getExecutorInfos'):
                    spark_executors = len(status_tracker.getExecutorInfos()) - 1  # Exclude driver
                # Method 2: Try getting from Spark UI metrics  
                elif hasattr(spark_context, '_jsc') and hasattr(spark_context._jsc, 'statusTracker'):
                    # Alternative approach through Java context
                    jvm_tracker = spark_context._jsc.statusTracker()
                    if hasattr(jvm_tracker, 'getExecutorInfos'):
                        spark_executors = len(jvm_tracker.getExecutorInfos()) - 1
                    else:
                        # Fallback: estimate from configuration
                        spark_executors = max(1, spark_context.defaultParallelism // 2)
                else:
                    # Final fallback: use configuration hints
                    spark_executors = max(1, spark_context.defaultParallelism // 2)
            except Exception as e:
                # Suppress the warning for this known compatibility issue
                # self.logger.warning(f"Resource monitoring error: {str(e)}")
                spark_executors = max(1, spark_context.defaultParallelism // 2)
            
            resource_metrics = {
                'timestamp': datetime.now(),
                'memory_usage_mb': (memory.total - memory.available) / (1024 * 1024),
                'memory_usage_percent': memory.percent,
                'memory_available_mb': memory.available / (1024 * 1024),
                'cpu_usage_percent': cpu_percent,
                'cpu_count': psutil.cpu_count(),
                'disk_read_mb': disk_io.read_bytes / (1024 * 1024) if disk_io else 0,
                'disk_write_mb': disk_io.write_bytes / (1024 * 1024) if disk_io else 0,
                'spark_executors': spark_executors,
                'spark_storage_memory_used': 0,  # Not available in this Spark version
                'spark_storage_memory_remaining': 0  # Not available in this Spark version
            }
            
            # Store sample
            self.resource_samples.append(resource_metrics)
            
            # Update peak metrics
            self.metrics.peak_memory_usage_mb = max(
                self.metrics.peak_memory_usage_mb, 
                resource_metrics['memory_usage_mb']
            )
            self.metrics.peak_cpu_usage_percent = max(
                self.metrics.peak_cpu_usage_percent,
                resource_metrics['cpu_usage_percent']
            )
            
            return resource_metrics
            
        except Exception as e:
            self.logger.warning(f"Resource monitoring error: {e}")
            return {}
    
    def _start_resource_monitoring(self):
        """Start background resource monitoring"""
        def monitor_resources():
            while self.monitoring_active:
                self.monitor_resource_usage()
                time.sleep(self.resource_monitoring_interval)
        
        # Start monitoring in background thread
        from threading import Thread
        monitor_thread = Thread(target=monitor_resources, daemon=True)
        monitor_thread.start()
    
    def _estimate_dataframe_size(self, df: DataFrame, sample_size: int = 1000) -> float:
        """
        Estimate DataFrame size in MB
        
        Args:
            df: Spark DataFrame
            sample_size: Number of rows to sample for estimation
            
        Returns:
            float: Estimated size in MB
        """
        try:
            # Get total row count
            total_rows = df.count()
            
            if total_rows == 0:
                return 0.0
            
            # Sample data and estimate size
            sample_df = df.limit(sample_size)
            sample_pandas = sample_df.toPandas()
            
            # Estimate size in bytes
            sample_size_bytes = sample_pandas.memory_usage(deep=True).sum()
            
            # Scale to total size
            estimated_size_bytes = (sample_size_bytes / len(sample_pandas)) * total_rows
            
            return estimated_size_bytes / (1024 * 1024)  # Convert to MB
            
        except Exception as e:
            self.logger.warning(f"Could not estimate DataFrame size: {e}")
            return 0.0
    
    def _calculate_performance_metrics(self):
        """Calculate aggregated performance metrics"""
        if not self.performance_samples:
            return
            
        # Calculate throughput
        total_records = sum(sample['records_processed'] for sample in self.performance_samples)
        total_duration = sum(sample['duration_seconds'] for sample in self.performance_samples)
        
        if total_duration > 0:
            self.metrics.throughput_records_per_second = total_records / total_duration
            self.metrics.avg_processing_time_per_record = (total_duration * 1000) / max(total_records, 1)
    
    def _calculate_resource_metrics(self):
        """Calculate aggregated resource metrics"""
        if not self.resource_samples:
            return
            
        memory_usage = [sample['memory_usage_mb'] for sample in self.resource_samples]
        cpu_usage = [sample['cpu_usage_percent'] for sample in self.resource_samples]
        
        if memory_usage:
            self.metrics.avg_memory_usage_mb = np.mean(memory_usage)
            
        if cpu_usage:
            self.metrics.avg_cpu_usage_percent = np.mean(cpu_usage)
            
        # Calculate I/O metrics
        if len(self.resource_samples) >= 2:
            first_sample = self.resource_samples[0]
            last_sample = self.resource_samples[-1]
            
            self.metrics.disk_io_read_mb = last_sample['disk_read_mb'] - first_sample['disk_read_mb']
            self.metrics.disk_io_write_mb = last_sample['disk_write_mb'] - first_sample['disk_write_mb']
    
    def _calculate_scalability_scores(self):
        """Calculate scalability scoring metrics"""
        # Linear scalability score (based on throughput efficiency)
        if self.metrics.total_rows_processed > 0 and self.metrics.total_processing_time > 0:
            # Perfect scalability baseline: 1000 records/second
            baseline_throughput = 1000
            actual_throughput = self.metrics.throughput_records_per_second
            
            self.metrics.linear_scalability_score = min(1.0, actual_throughput / baseline_throughput)
        
        # Resource efficiency score
        if self.metrics.peak_memory_usage_mb > 0 and self.metrics.peak_cpu_usage_percent > 0:
            # Optimal: 70-80% resource utilization
            memory_efficiency = self._calculate_utilization_score(self.metrics.avg_memory_usage_mb / 1024, 70, 80)  # Convert to GB
            cpu_efficiency = self._calculate_utilization_score(self.metrics.avg_cpu_usage_percent, 60, 80)
            
            self.metrics.resource_efficiency_score = (memory_efficiency + cpu_efficiency) / 2
        
        # Cost efficiency (simplified model)
        if self.metrics.total_processing_time > 0 and self.metrics.total_rows_processed > 0:
            # Assume cost based on processing time and resource usage
            cost_per_hour = 1.0  # $1/hour baseline
            processing_hours = self.metrics.total_processing_time / 3600
            records_per_dollar = self.metrics.total_rows_processed / (cost_per_hour * processing_hours)
            
            # Normalize to 0-1 scale (baseline: 10000 records per dollar)
            self.metrics.cost_efficiency_score = min(1.0, records_per_dollar / 10000)
    
    def _calculate_utilization_score(self, actual: float, optimal_min: float, optimal_max: float) -> float:
        """Calculate utilization efficiency score"""
        if actual < optimal_min:
            return actual / optimal_min  # Underutilization penalty
        elif actual > optimal_max:
            return max(0.1, (100 - actual) / (100 - optimal_max))  # Overutilization penalty
        else:
            return 1.0  # Optimal range
    
    def _calculate_partition_efficiency(self, partition_stats: List[Dict]) -> float:
        """Calculate partition efficiency score"""
        if not partition_stats:
            return 0.0
            
        rows_per_partition = [p['rows_per_partition'] for p in partition_stats]
        
        if not rows_per_partition:
            return 0.0
            
        # Calculate coefficient of variation (lower is better)
        mean_rows = np.mean(rows_per_partition)
        std_rows = np.std(rows_per_partition)
        
        if mean_rows == 0:
            return 0.0
            
        coefficient_of_variation = std_rows / mean_rows
        
        # Convert to efficiency score (0-1, higher is better)
        return max(0.0, 1.0 - min(coefficient_of_variation, 1.0))
    
    def _identify_bottlenecks(self):
        """Identify potential scalability bottlenecks"""
        bottlenecks = {}
        
        # Memory bottlenecks
        if self.metrics.peak_memory_usage_mb > 0:
            memory_pressure = self.metrics.avg_memory_usage_mb / self.metrics.peak_memory_usage_mb
            if memory_pressure > 0.9:
                bottlenecks['memory_pressure'] = {
                    'severity': 'high',
                    'description': f'High memory usage: {memory_pressure:.1%}',
                    'recommendation': 'Consider increasing executor memory or optimizing data structures'
                }
        
        # CPU bottlenecks
        if self.metrics.avg_cpu_usage_percent > 85:
            bottlenecks['cpu_saturation'] = {
                'severity': 'medium',
                'description': f'High CPU usage: {self.metrics.avg_cpu_usage_percent:.1f}%',
                'recommendation': 'Consider adding more CPU cores or optimizing algorithms'
            }
        
        # Throughput bottlenecks
        if self.metrics.throughput_records_per_second < 100:  # Arbitrary threshold
            bottlenecks['low_throughput'] = {
                'severity': 'high',
                'description': f'Low throughput: {self.metrics.throughput_records_per_second:.1f} records/sec',
                'recommendation': 'Optimize data processing or increase parallelism'
            }
        
        # Partition inefficiency
        if self.metrics.partition_efficiency_score < 0.7:
            bottlenecks['partition_skew'] = {
                'severity': 'medium',
                'description': f'Partition inefficiency: {self.metrics.partition_efficiency_score:.2f}',
                'recommendation': 'Consider repartitioning data for better load distribution'
            }
        
        self.metrics.bottleneck_indicators = bottlenecks
    
    def _log_metrics_to_mlflow(self):
        """Log scalability metrics to MLflow"""
        try:
            with mlflow.start_run(run_name=f"scalability_monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Set tags
                mlflow.set_tag("monitoring_type", "scalability")
                mlflow.set_tag("pipeline_version", "1.0")
                
                # Data volume metrics
                mlflow.log_metric("total_rows_processed", self.metrics.total_rows_processed)
                mlflow.log_metric("data_size_mb", self.metrics.data_size_mb)
                mlflow.log_metric("partition_efficiency_score", self.metrics.partition_efficiency_score)
                
                # Performance metrics
                mlflow.log_metric("total_processing_time", self.metrics.total_processing_time)
                mlflow.log_metric("throughput_records_per_second", self.metrics.throughput_records_per_second)
                mlflow.log_metric("avg_processing_time_per_record", self.metrics.avg_processing_time_per_record)
                
                # Resource metrics
                mlflow.log_metric("peak_memory_usage_mb", self.metrics.peak_memory_usage_mb)
                mlflow.log_metric("avg_memory_usage_mb", self.metrics.avg_memory_usage_mb)
                mlflow.log_metric("peak_cpu_usage_percent", self.metrics.peak_cpu_usage_percent)
                mlflow.log_metric("avg_cpu_usage_percent", self.metrics.avg_cpu_usage_percent)
                
                # Scalability scores
                mlflow.log_metric("linear_scalability_score", self.metrics.linear_scalability_score)
                mlflow.log_metric("resource_efficiency_score", self.metrics.resource_efficiency_score)
                mlflow.log_metric("cost_efficiency_score", self.metrics.cost_efficiency_score)
                
                # Log bottlenecks as parameters
                for bottleneck, details in self.metrics.bottleneck_indicators.items():
                    mlflow.log_param(f"bottleneck_{bottleneck}", details['severity'])
                
                self.logger.info("✅ Scalability metrics logged to MLflow")
                
        except Exception as e:
            self.logger.warning(f"Could not log to MLflow: {e}")
    
    def generate_monitoring_report(self, output_path: str = None) -> str:
        """
        Generate comprehensive monitoring report
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            str: Report content
        """
        report_lines = [
            "=" * 80,
            "🔍 SCALABILITY MONITORING REPORT",
            "=" * 80,
            f"Monitoring Period: {self.metrics.start_time} to {self.metrics.end_time}",
            f"Total Processing Time: {self.metrics.total_processing_time:.2f} seconds",
            "",
            "📊 DATA VOLUME METRICS:",
            "-" * 40,
            f"Total Rows Processed: {self.metrics.total_rows_processed:,}",
            f"Data Size: {self.metrics.data_size_mb:.1f} MB",
            f"Partition Count: {self.metrics.partition_count}",
            f"Avg Partition Size: {self.metrics.avg_partition_size_mb:.1f} MB",
            f"Partition Efficiency: {self.metrics.partition_efficiency_score:.3f}",
            "",
            "⚡ PERFORMANCE METRICS:",
            "-" * 40,
            f"Throughput: {self.metrics.throughput_records_per_second:.1f} records/sec",
            f"Avg Time per Record: {self.metrics.avg_processing_time_per_record:.2f} ms",
            f"Feature Engineering Time: {self.metrics.feature_engineering_time:.2f} sec",
            f"Model Inference Time: {self.metrics.model_inference_time:.2f} sec",
            "",
            "💾 RESOURCE UTILIZATION:",
            "-" * 40,
            f"Peak Memory Usage: {self.metrics.peak_memory_usage_mb:.1f} MB",
            f"Avg Memory Usage: {self.metrics.avg_memory_usage_mb:.1f} MB",
            f"Peak CPU Usage: {self.metrics.peak_cpu_usage_percent:.1f}%",
            f"Avg CPU Usage: {self.metrics.avg_cpu_usage_percent:.1f}%",
            f"Disk I/O Read: {self.metrics.disk_io_read_mb:.1f} MB",
            f"Disk I/O Write: {self.metrics.disk_io_write_mb:.1f} MB",
            "",
            "🎯 SCALABILITY SCORES:",
            "-" * 40,
            f"Linear Scalability Score: {self.metrics.linear_scalability_score:.3f}",
            f"Resource Efficiency Score: {self.metrics.resource_efficiency_score:.3f}",
            f"Cost Efficiency Score: {self.metrics.cost_efficiency_score:.3f}",
            "",
        ]
        
        # Add bottleneck analysis
        if self.metrics.bottleneck_indicators:
            report_lines.extend([
                "⚠️  BOTTLENECK ANALYSIS:",
                "-" * 40,
            ])
            
            for bottleneck, details in self.metrics.bottleneck_indicators.items():
                severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(details['severity'], "ℹ️")
                report_lines.extend([
                    f"{severity_emoji} {bottleneck.upper()}:",
                    f"  {details['description']}",
                    f"  Recommendation: {details['recommendation']}",
                    ""
                ])
        else:
            report_lines.extend([
                "✅ BOTTLENECK ANALYSIS:",
                "-" * 40,
                "No significant bottlenecks detected!",
                ""
            ])
        
        report_lines.append("=" * 80)
        
        report_content = "\n".join(report_lines)
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_content)
            self.logger.info(f"📄 Monitoring report saved to: {output_path}")
        
        return report_content
    
    def export_metrics_json(self, output_path: str) -> None:
        """Export metrics to JSON file"""
        metrics_dict = asdict(self.metrics)
        
        # Convert datetime objects to strings
        for key, value in metrics_dict.items():
            if isinstance(value, datetime):
                metrics_dict[key] = value.isoformat()
        
        with open(output_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2, default=str)
        
        self.logger.info(f"📊 Metrics exported to: {output_path}")


class PerformanceBenchmark:
    """Benchmark suite for testing scalability under different conditions"""
    
    def __init__(self, config: Config, spark: SparkSession):
        self.config = config
        self.spark = spark
        self.logger = Logger().get_logger()
    
    def run_data_volume_benchmark(self, multipliers: List[int] = [1, 2, 5, 10]) -> Dict:
        """
        Benchmark performance across different data volumes
        
        Args:
            multipliers: Data volume multipliers to test
            
        Returns:
            Dict: Benchmark results
        """
        results = {}
        
        for multiplier in multipliers:
            self.logger.info(f"🏃 Running benchmark with {multiplier}x data volume")
            
            # Create synthetic data for testing
            benchmark_data = self._create_benchmark_data(multiplier)
            
            # Monitor processing
            monitor = ScalabilityMonitor(self.config, self.spark)
            monitor.start_monitoring(f"benchmark_{multiplier}x")
            
            start_time = time.time()
            
            # Simulate data processing
            processed_rows = self._process_benchmark_data(benchmark_data)
            
            performance = monitor.monitor_performance("data_processing", start_time, processed_rows)
            metrics = monitor.stop_monitoring()
            
            results[f"{multiplier}x"] = {
                'data_volume_multiplier': multiplier,
                'processed_rows': processed_rows,
                'processing_time': performance['duration_seconds'],
                'throughput': performance['throughput_records_per_second'],
                'scalability_metrics': asdict(metrics)
            }
            
            self.logger.info(f"✅ Benchmark {multiplier}x completed: {performance['throughput_records_per_second']:.1f} records/sec")
        
        return results
    
    def _create_benchmark_data(self, multiplier: int) -> DataFrame:
        """Create synthetic benchmark data"""
        # Create a simple DataFrame for testing
        rows = 1000 * multiplier
        
        data = [(i, f"stock_{i % 5}", float(i * 1.5), float(i * 2.0)) 
                for i in range(rows)]
        
        schema = StructType([
            StructField("id", IntegerType(), False),
            StructField("symbol", StringType(), False),
            StructField("price", DoubleType(), False),
            StructField("volume", DoubleType(), False)
        ])
        
        return self.spark.createDataFrame(data, schema)
    
    def _process_benchmark_data(self, df: DataFrame) -> int:
        """Process benchmark data to simulate real workload"""
        # Simulate feature engineering operations
        df_processed = df.withColumn("price_squared", col("price") * col("price"))
        df_processed = df_processed.withColumn("volume_log", col("volume"))
        
        # Force computation
        count = df_processed.count()
        
        return count