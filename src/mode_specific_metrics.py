#!/usr/bin/env python3
"""
Mode-Specific Performance Metrics Module
Specialized metrics collection for each training mode: Spark GBT, Spark RF, Hybrid Sklearn

Part of Data Engineering at Scale project - Phase 3 implementation
"""

import time
import psutil
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count
import pyspark.sql.functions as F

from src.utils import Logger


@dataclass
class SparkGBTMetrics:
    """Metrics specific to Spark GBT mode (driver-focused training)"""
    
    # Driver-specific metrics
    driver_memory_usage_mb: float = 0.0
    driver_cpu_usage_percent: float = 0.0
    driver_gc_time_ms: float = 0.0
    
    # Sequential training metrics
    sequential_training_time: float = 0.0
    tree_building_time: float = 0.0
    gradient_computation_time: float = 0.0
    
    # Model-specific metrics
    max_tree_depth_achieved: int = 0
    total_trees_built: int = 0
    feature_importance_computation_time: float = 0.0
    
    # Memory pressure indicators
    driver_memory_pressure_warnings: int = 0
    gc_overhead_percentage: float = 0.0


@dataclass
class SparkRFMetrics:
    """Metrics specific to Spark RF mode (distributed training)"""
    
    # Distributed training metrics
    executor_utilization_percent: float = 0.0
    parallel_tree_building_efficiency: float = 0.0
    inter_executor_communication_time: float = 0.0
    
    # Forest-specific metrics
    trees_per_executor: Dict[str, int] = None
    distributed_training_coordination_time: float = 0.0
    ensemble_aggregation_time: float = 0.0
    
    # Parallelism efficiency
    theoretical_speedup: float = 0.0
    actual_speedup: float = 0.0
    parallelism_efficiency_score: float = 0.0
    
    # Resource distribution
    task_distribution_balance: float = 0.0
    executor_memory_balance: float = 0.0
    
    def __post_init__(self):
        if self.trees_per_executor is None:
            self.trees_per_executor = {}


@dataclass
class HybridSklearnMetrics:
    """Metrics specific to Hybrid Sklearn mode"""
    
    # Data transfer metrics
    spark_to_driver_transfer_time: float = 0.0
    data_collection_size_mb: float = 0.0
    transfer_throughput_mbps: float = 0.0
    
    # Driver sklearn training metrics
    sklearn_training_time: float = 0.0
    driver_memory_during_sklearn: float = 0.0
    sklearn_model_size_mb: float = 0.0
    
    # Distributed inference metrics
    pandas_udf_setup_time: float = 0.0
    distributed_inference_time: float = 0.0
    inference_task_distribution_time: float = 0.0
    
    # Hybrid efficiency metrics
    data_transfer_overhead_percentage: float = 0.0
    hybrid_vs_spark_overhead: float = 0.0
    memory_driver_vs_executor_ratio: float = 0.0
    
    # Model serialization metrics
    model_serialization_time: float = 0.0
    model_deserialization_time: float = 0.0


class ModeSpecificMonitor:
    """
    Mode-specific performance monitor for different training approaches
    """
    
    def __init__(self, mode: str, spark: SparkSession, config: Dict = None):
        """
        Initialize mode-specific monitor
        
        Args:
            mode: Training mode ('spark_gbt', 'spark_rf', 'hybrid_sklearn')
            spark: Spark session
            config: Optional configuration
        """
        self.mode = mode
        self.spark = spark
        self.config = config or {}
        self.logger = Logger().get_logger()
        
        # Initialize metrics based on mode
        self.metrics = self._initialize_metrics()
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.start_time = None
        
        # Data collection
        self.resource_samples = []
        self.performance_checkpoints = []
        
        # Performance thresholds based on data size (5k-80k rows, scalable to 500k-1M)
        self.performance_thresholds = self._calculate_thresholds()
        
        self.logger.info(f"🔍 Mode-specific monitor initialized for: {mode}")
    
    def _initialize_metrics(self) -> Any:
        """Initialize metrics object based on mode"""
        if self.mode == "spark_gbt":
            return SparkGBTMetrics()
        elif self.mode == "spark_rf":
            return SparkRFMetrics()
        elif self.mode == "hybrid_sklearn":
            return HybridSklearnMetrics()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _calculate_thresholds(self) -> Dict:
        """Calculate performance thresholds based on data size and system config"""
        
        # Get system specs
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Base thresholds for small dataset (5k rows)
        base_thresholds = {
            "training_time_per_1k_rows_ms": 100,  # 100ms per 1k rows
            "memory_usage_per_1k_rows_mb": 10,    # 10MB per 1k rows
            "cost_efficiency_target": 0.8,        # Memory×Time efficiency target
        }
        
        # Scale factors based on system capacity
        cpu_scale_factor = min(cpu_count / 8, 2.0)  # Cap at 2x improvement for 8+ cores
        memory_scale_factor = min(memory_gb / 16, 2.0)  # Cap at 2x improvement for 16+ GB
        
        # Calculate thresholds
        thresholds = {
            "max_training_time_per_1k_rows_ms": base_thresholds["training_time_per_1k_rows_ms"] / cpu_scale_factor,
            "max_memory_per_1k_rows_mb": base_thresholds["memory_usage_per_1k_rows_mb"] / memory_scale_factor,
            "min_cost_efficiency": base_thresholds["cost_efficiency_target"],
            
            # Mode-specific thresholds
            "spark_gbt": {
                "max_driver_memory_pressure": 0.85,  # 85% driver memory usage
                "max_gc_overhead_percentage": 20.0,   # 20% GC overhead
            },
            "spark_rf": {
                "min_parallelism_efficiency": 0.7,   # 70% parallelism efficiency
                "min_executor_utilization": 0.8,     # 80% executor utilization
            },
            "hybrid_sklearn": {
                "max_data_transfer_overhead": 0.3,   # 30% transfer overhead
                "min_hybrid_efficiency": 0.6,        # 60% hybrid efficiency
            }
        }
        
        self.logger.info(f"📊 Performance thresholds calculated for {cpu_count} CPUs, {memory_gb:.1f}GB RAM")
        return thresholds
    
    def start_monitoring(self, run_name: str = None):
        """Start mode-specific monitoring"""
        if self.monitoring_active:
            self.logger.warning("⚠️ Monitoring already active")
            return
        
        self.monitoring_active = True
        self.start_time = time.time()
        
        # Start background monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        self.logger.info(f"🔍 Started {self.mode} monitoring: {run_name or 'unnamed_run'}")
    
    def stop_monitoring(self) -> Dict:
        """Stop monitoring and return collected metrics"""
        if not self.monitoring_active:
            self.logger.warning("⚠️ Monitoring not active")
            return {}
        
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        
        # Calculate final metrics
        self._calculate_final_metrics()
        
        # Evaluate against thresholds
        threshold_evaluation = self._evaluate_thresholds()
        
        result = {
            "mode": self.mode,
            "metrics": asdict(self.metrics),
            "resource_samples": self.resource_samples,
            "performance_checkpoints": self.performance_checkpoints,
            "threshold_evaluation": threshold_evaluation,
            "monitoring_duration": time.time() - self.start_time
        }
        
        self.logger.info(f"✅ Stopped {self.mode} monitoring")
        return result
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system resources
                self._collect_system_metrics()
                
                # Collect Spark metrics if available
                self._collect_spark_metrics()
                
                # Mode-specific monitoring
                if self.mode == "spark_gbt":
                    self._monitor_spark_gbt()
                elif self.mode == "spark_rf":
                    self._monitor_spark_rf()
                elif self.mode == "hybrid_sklearn":
                    self._monitor_hybrid_sklearn()
                
                time.sleep(1.0)  # Monitor every second
                
            except Exception as e:
                self.logger.error(f"❌ Monitoring error: {e}")
    
    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            
            sample = {
                "timestamp": time.time(),
                "memory_used_mb": (memory.total - memory.available) / (1024**2),
                "memory_percent": memory.percent,
                "cpu_percent": cpu_percent,
                "disk_read_mb": disk_io.read_bytes / (1024**2) if disk_io else 0,
                "disk_write_mb": disk_io.write_bytes / (1024**2) if disk_io else 0,
            }
            
            self.resource_samples.append(sample)
            
            # Update metrics based on mode
            if self.mode in ["spark_gbt", "hybrid_sklearn"]:
                # Driver-focused modes
                if hasattr(self.metrics, 'driver_memory_usage_mb'):
                    self.metrics.driver_memory_usage_mb = max(
                        self.metrics.driver_memory_usage_mb,
                        sample["memory_used_mb"]
                    )
                if hasattr(self.metrics, 'driver_cpu_usage_percent'):
                    self.metrics.driver_cpu_usage_percent = max(
                        self.metrics.driver_cpu_usage_percent,
                        sample["cpu_percent"]
                    )
        
        except Exception as e:
            self.logger.error(f"❌ System metrics collection error: {e}")
    
    def _collect_spark_metrics(self):
        """Collect Spark-specific metrics"""
        try:
            # Get Spark context
            sc = self.spark.sparkContext
            status_tracker = sc.statusTracker()
            
            # Collect executor metrics if available
            executor_metrics = {}
            try:
                # Try to get executor information (may not be available in local mode)
                if hasattr(status_tracker, 'getExecutorInfos'):
                    executors = status_tracker.getExecutorInfos()
                    executor_metrics = {
                        "executor_count": len(executors),
                        "total_cores": sum(exec.totalCores for exec in executors),
                        "total_memory_mb": sum(exec.maxMemory / (1024**2) for exec in executors)
                    }
            except Exception:
                # Fallback for local mode
                executor_metrics = {
                    "executor_count": 1,
                    "total_cores": psutil.cpu_count(),
                    "total_memory_mb": psutil.virtual_memory().total / (1024**2)
                }
            
            # Update RF metrics if applicable
            if self.mode == "spark_rf" and hasattr(self.metrics, 'executor_utilization_percent'):
                active_jobs = len(status_tracker.getActiveJobIds())
                # Estimate utilization based on active jobs
                self.metrics.executor_utilization_percent = min(100.0, active_jobs * 25.0)
        
        except Exception as e:
            self.logger.debug(f"Spark metrics collection: {e}")  # Debug level since this might fail in local mode
    
    def _monitor_spark_gbt(self):
        """Monitor Spark GBT specific metrics"""
        try:
            # Monitor garbage collection if available
            import gc
            gc_stats = gc.get_stats()
            if gc_stats:
                # Estimate GC overhead
                total_collections = sum(stat.get('collections', 0) for stat in gc_stats)
                if hasattr(self.metrics, 'driver_gc_time_ms'):
                    self.metrics.driver_gc_time_ms += total_collections * 10  # Rough estimate
        
        except Exception as e:
            self.logger.debug(f"GBT monitoring: {e}")
    
    def _monitor_spark_rf(self):
        """Monitor Spark RF specific metrics"""
        try:
            # Calculate theoretical vs actual speedup
            if hasattr(self.metrics, 'theoretical_speedup'):
                cpu_count = psutil.cpu_count()
                self.metrics.theoretical_speedup = min(cpu_count, 8)  # Cap at reasonable level
                
                # Estimate actual speedup based on CPU usage
                if self.resource_samples:
                    recent_cpu = self.resource_samples[-1]["cpu_percent"]
                    self.metrics.actual_speedup = recent_cpu / 100.0 * cpu_count
                    
                    # Calculate parallelism efficiency
                    if self.metrics.theoretical_speedup > 0:
                        self.metrics.parallelism_efficiency_score = min(1.0,
                            self.metrics.actual_speedup / self.metrics.theoretical_speedup
                        )
        
        except Exception as e:
            self.logger.debug(f"RF monitoring: {e}")
    
    def _monitor_hybrid_sklearn(self):
        """Monitor Hybrid Sklearn specific metrics"""
        try:
            # Monitor memory pressure for driver vs executor balance
            if self.resource_samples:
                recent_memory = self.resource_samples[-1]["memory_used_mb"]
                total_memory = psutil.virtual_memory().total / (1024**2)
                
                if hasattr(self.metrics, 'memory_driver_vs_executor_ratio'):
                    # Estimate driver memory usage ratio
                    self.metrics.memory_driver_vs_executor_ratio = recent_memory / total_memory
        
        except Exception as e:
            self.logger.debug(f"Hybrid monitoring: {e}")
    
    def _calculate_final_metrics(self):
        """Calculate final aggregated metrics"""
        if not self.resource_samples:
            return
        
        # Calculate aggregated system metrics
        total_duration = time.time() - self.start_time
        
        # Memory efficiency calculation
        memory_values = [sample["memory_used_mb"] for sample in self.resource_samples]
        avg_memory = sum(memory_values) / len(memory_values)
        peak_memory = max(memory_values)
        
        # CPU utilization
        cpu_values = [sample["cpu_percent"] for sample in self.resource_samples]
        avg_cpu = sum(cpu_values) / len(cpu_values)
        peak_cpu = max(cpu_values)
        
        # Update common metrics across all modes
        for mode_metrics in [self.metrics]:
            if hasattr(mode_metrics, 'avg_memory_usage_mb'):
                mode_metrics.avg_memory_usage_mb = avg_memory
            if hasattr(mode_metrics, 'peak_memory_usage_mb'):
                mode_metrics.peak_memory_usage_mb = peak_memory
            if hasattr(mode_metrics, 'avg_cpu_usage_percent'):
                mode_metrics.avg_cpu_usage_percent = avg_cpu
            if hasattr(mode_metrics, 'peak_cpu_usage_percent'):
                mode_metrics.peak_cpu_usage_percent = peak_cpu
    
    def _evaluate_thresholds(self) -> Dict:
        """Evaluate metrics against performance thresholds"""
        evaluation = {
            "overall_status": "PASS",
            "warnings": [],
            "failures": [],
            "performance_score": 1.0
        }
        
        try:
            thresholds = self.performance_thresholds
            
            # Mode-specific threshold evaluation
            if self.mode == "spark_gbt":
                if hasattr(self.metrics, 'driver_memory_usage_mb'):
                    memory_gb = psutil.virtual_memory().total / (1024**3)
                    memory_pressure = self.metrics.driver_memory_usage_mb / (memory_gb * 1024)
                    
                    if memory_pressure > thresholds["spark_gbt"]["max_driver_memory_pressure"]:
                        evaluation["warnings"].append(
                            f"High driver memory pressure: {memory_pressure:.2f}"
                        )
            
            elif self.mode == "spark_rf":
                if hasattr(self.metrics, 'parallelism_efficiency_score'):
                    if self.metrics.parallelism_efficiency_score < thresholds["spark_rf"]["min_parallelism_efficiency"]:
                        evaluation["warnings"].append(
                            f"Low parallelism efficiency: {self.metrics.parallelism_efficiency_score:.2f}"
                        )
            
            elif self.mode == "hybrid_sklearn":
                if hasattr(self.metrics, 'data_transfer_overhead_percentage'):
                    if self.metrics.data_transfer_overhead_percentage > thresholds["hybrid_sklearn"]["max_data_transfer_overhead"]:
                        evaluation["warnings"].append(
                            f"High data transfer overhead: {self.metrics.data_transfer_overhead_percentage:.2f}%"
                        )
            
            # Overall evaluation
            evaluation["overall_status"] = "FAIL" if evaluation["failures"] else (
                "WARN" if evaluation["warnings"] else "PASS"
            )
            
            # Calculate performance score (0.0 to 1.0)
            penalty_factor = len(evaluation["warnings"]) * 0.1 + len(evaluation["failures"]) * 0.3
            evaluation["performance_score"] = max(0.0, 1.0 - penalty_factor)
        
        except Exception as e:
            self.logger.error(f"❌ Threshold evaluation error: {e}")
            evaluation["overall_status"] = "ERROR"
            evaluation["failures"].append(f"Evaluation error: {e}")
        
        return evaluation


def create_mode_monitor(mode: str, spark: SparkSession, config: Dict = None) -> ModeSpecificMonitor:
    """Factory function to create mode-specific monitor"""
    return ModeSpecificMonitor(mode, spark, config)


# Testing functions for validation
def test_mode_monitor_spark_gbt(spark: SparkSession):
    """Test GBT mode monitoring"""
    logger = Logger().get_logger()
    logger.info("🧪 Testing Spark GBT mode monitor...")
    
    monitor = create_mode_monitor("spark_gbt", spark)
    monitor.start_monitoring("test_gbt_run")
    
    # Simulate some work
    time.sleep(3)
    
    result = monitor.stop_monitoring()
    
    # Validate results
    assert result["mode"] == "spark_gbt"
    assert "metrics" in result
    assert len(result["resource_samples"]) > 0
    
    logger.info("✅ Spark GBT monitor test passed")
    return result


def test_mode_monitor_spark_rf(spark: SparkSession):
    """Test RF mode monitoring"""
    logger = Logger().get_logger()
    logger.info("🧪 Testing Spark RF mode monitor...")
    
    monitor = create_mode_monitor("spark_rf", spark)
    monitor.start_monitoring("test_rf_run")
    
    # Simulate some work
    time.sleep(3)
    
    result = monitor.stop_monitoring()
    
    # Validate results
    assert result["mode"] == "spark_rf"
    assert "metrics" in result
    assert len(result["resource_samples"]) > 0
    
    logger.info("✅ Spark RF monitor test passed")
    return result


def test_mode_monitor_hybrid_sklearn(spark: SparkSession):
    """Test Hybrid Sklearn mode monitoring"""
    logger = Logger().get_logger()
    logger.info("🧪 Testing Hybrid Sklearn mode monitor...")
    
    monitor = create_mode_monitor("hybrid_sklearn", spark)
    monitor.start_monitoring("test_hybrid_run")
    
    # Simulate some work
    time.sleep(3)
    
    result = monitor.stop_monitoring()
    
    # Validate results
    assert result["mode"] == "hybrid_sklearn"
    assert "metrics" in result
    assert len(result["resource_samples"]) > 0
    
    logger.info("✅ Hybrid Sklearn monitor test passed")
    return result