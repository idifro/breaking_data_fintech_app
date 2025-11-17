#!/usr/bin/env python3
"""
Optimized Spark Session Creation for ML Training
Enhanced for local development with system resource monitoring
"""

import os
import psutil
import logging
from typing import Dict, Optional
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip


class SparkResourceManager:
    """Manages Spark configuration based on system resources"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.system_info = self._get_system_info()
    
    def _get_system_info(self) -> Dict:
        """Get current system resource information"""
        cpu_count = psutil.cpu_count(logical=False)  # Physical cores
        logical_cpu_count = psutil.cpu_count(logical=True)
        memory = psutil.virtual_memory()
        
        return {
            'physical_cores': cpu_count,
            'logical_cores': logical_cpu_count,
            'total_memory_gb': memory.total / (1024**3),
            'available_memory_gb': memory.available / (1024**3),
            'memory_percent_used': memory.percent
        }
    
    def get_optimal_spark_config(self, workload_type: str = "ml_training") -> Dict:
        """Get optimal Spark configuration based on system resources and workload"""
        
        config = {}
        sys_info = self.system_info
        
        # Memory allocation (leave 20-25% for OS)
        available_memory = sys_info['available_memory_gb']
        reserved_memory = max(4, available_memory * 0.25)  # Reserve at least 4GB or 25%
        usable_memory = available_memory - reserved_memory
        
        if workload_type == "ml_training":
            # For ML workloads, allocate more to driver for feature engineering
            driver_memory = min(12, usable_memory * 0.6)
            executor_memory = min(10, usable_memory * 0.4)
            
            # Use physical cores for better performance
            executor_cores = min(4, sys_info['physical_cores'])
            
            config.update({
                'spark.driver.memory': f'{driver_memory:.0f}g',
                'spark.executor.memory': f'{executor_memory:.0f}g',
                'spark.executor.cores': str(executor_cores),
                'spark.master': f'local[{sys_info["physical_cores"]}]',
                
                # ML-specific configurations
                'spark.ml.cache.enabled': 'true',
                'spark.mllib.cache.enabled': 'true',
                'spark.sql.adaptive.advisoryPartitionSizeInBytes': '32MB',
                'spark.sql.adaptive.maxNumPostShufflePartitions': '64',
                'spark.sql.execution.arrow.maxRecordsPerBatch': '10000',
            })
            
        elif workload_type == "data_processing":
            # For data processing, balance driver and executor memory
            driver_memory = min(8, usable_memory * 0.4)
            executor_memory = min(12, usable_memory * 0.6)
            
            config.update({
                'spark.driver.memory': f'{driver_memory:.0f}g',
                'spark.executor.memory': f'{executor_memory:.0f}g',
                'spark.executor.cores': '4',
                'spark.master': f'local[{sys_info["logical_cores"]}]',
                
                # Data processing optimizations
                'spark.sql.adaptive.advisoryPartitionSizeInBytes': '64MB',
                'spark.sql.adaptive.maxNumPostShufflePartitions': '200',
            })
        
        return config
    
    def log_system_info(self):
        """Log system information for monitoring"""
        sys_info = self.system_info
        self.logger.info("🖥️  System Resources:")
        self.logger.info(f"   CPU: {sys_info['physical_cores']} cores ({sys_info['logical_cores']} logical)")
        self.logger.info(f"   Memory: {sys_info['total_memory_gb']:.1f}GB total, {sys_info['available_memory_gb']:.1f}GB available")
        self.logger.info(f"   Memory usage: {sys_info['memory_percent_used']:.1f}%")


def create_optimized_spark_session(
    config, 
    workload_type: str = "ml_training",
    enable_monitoring: bool = True
) -> SparkSession:
    """
    Create optimized Spark session based on system resources and workload type
    
    Args:
        config: Application configuration object
        workload_type: Type of workload ('ml_training', 'data_processing')
        enable_monitoring: Whether to enable resource monitoring
    
    Returns:
        Configured SparkSession
    """
    
    # Initialize resource manager
    resource_manager = SparkResourceManager()
    
    if enable_monitoring:
        resource_manager.log_system_info()
    
    # Get optimal configuration
    optimal_config = resource_manager.get_optimal_spark_config(workload_type)
    
    # Create Spark session builder
    app_name = f"{config.spark.app_name}_{workload_type.title()}"
    builder = SparkSession.builder.appName(app_name)
    
    # Apply base configuration
    builder = builder \
        .master(optimal_config.get('spark.master', config.spark.master)) \
        .config("spark.driver.memory", optimal_config.get('spark.driver.memory', config.spark.driver_memory)) \
        .config("spark.executor.memory", optimal_config.get('spark.executor.memory', config.spark.executor_memory)) \
        .config("spark.executor.cores", optimal_config.get('spark.executor.cores', config.spark.executor_cores))
    
    # Add Delta Lake configuration
    builder = builder \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    
    # Add serialization and compression
    builder = builder \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.io.compression.codec", "snappy") \
        .config("spark.rdd.compress", "true")
    
    # Add Adaptive Query Execution
    builder = builder \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.adaptive.skewJoin.enabled", "true") \
        .config("spark.sql.adaptive.localShuffleReader.enabled", "true")
    
    # Add optimal configuration overrides
    for key, value in optimal_config.items():
        if key.startswith('spark.'):
            builder = builder.config(key, value)
    
    # Add additional configurations from config file
    for key, value in config.spark.configs.items():
        if key not in optimal_config:  # Don't override optimal settings
            builder = builder.config(key, value)
    
    # Add ML-specific optimizations for training workloads
    if workload_type == "ml_training":
        builder = builder \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.driver.maxResultSize", "4g") \
            .config("spark.driver.memoryFraction", "0.8") \
            .config("spark.executor.memoryFraction", "0.8")
    
    # Add GC optimization for local development
    gc_options = (
        "-XX:+UseG1GC "
        "-XX:MaxGCPauseMillis=200 "
        "-XX:InitiatingHeapOccupancyPercent=35"
    )
    
    builder = builder \
        .config("spark.driver.extraJavaOptions", gc_options) \
        .config("spark.executor.extraJavaOptions", gc_options)
    
    # Create session with Delta Lake support
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    
    # Set log level to reduce noise
    spark.sparkContext.setLogLevel("WARN")
    
    if enable_monitoring:
        # Log final configuration
        logger = logging.getLogger(__name__)
        logger.info("🚀 Spark Session Created Successfully")
        logger.info(f"   App Name: {app_name}")
        logger.info(f"   Master: {spark.sparkContext.master}")
        logger.info(f"   Driver Memory: {spark.conf.get('spark.driver.memory')}")
        logger.info(f"   Executor Memory: {spark.conf.get('spark.executor.memory')}")
        logger.info(f"   Executor Cores: {spark.conf.get('spark.executor.cores')}")
        logger.info(f"   Default Parallelism: {spark.sparkContext.defaultParallelism}")
    
    return spark


# Enhanced version of your original function
def create_spark_session(config, enable_monitoring: bool = True) -> SparkSession:
    """
    Enhanced version of your original create_spark_session function
    Now with automatic resource optimization
    """
    return create_optimized_spark_session(
        config, 
        workload_type="ml_training",
        enable_monitoring=enable_monitoring
    )


def monitor_spark_performance(spark: SparkSession) -> Dict:
    """Monitor current Spark session performance metrics"""
    
    # Get Spark context
    sc = spark.sparkContext
    
    # Get status tracker
    status_tracker = sc.statusTracker()
    
    # Collect metrics
    metrics = {
        'application_id': sc.applicationId,
        'application_name': sc.appName,
        'master': sc.master,
        'default_parallelism': sc.defaultParallelism,
        'executor_infos': len(status_tracker.getExecutorInfos()),
        'active_stages': len(status_tracker.getActiveStageIds()),
        'active_jobs': len(status_tracker.getActiveJobIds()),
    }
    
    # Get system metrics
    system_metrics = {
        'cpu_usage_percent': psutil.cpu_percent(interval=1),
        'memory_usage': psutil.virtual_memory(),
        'disk_usage': psutil.disk_usage('/tmp')
    }
    
    metrics.update(system_metrics)
    
    return metrics


if __name__ == "__main__":
    # Example usage
    import yaml
    from dataclasses import dataclass
    
    @dataclass
    class MockSparkConfig:
        app_name: str = "TestApp"
        master: str = "local[*]"
        driver_memory: str = "4g"
        executor_memory: str = "8g"
        executor_cores: str = "4"
        configs: dict = None
        
        def __post_init__(self):
            if self.configs is None:
                self.configs = {}
    
    @dataclass 
    class MockConfig:
        spark: MockSparkConfig = None
        
        def __post_init__(self):
            if self.spark is None:
                self.spark = MockSparkConfig()
    
    # Test the optimized session creation
    config = MockConfig()
    spark = create_optimized_spark_session(config, "ml_training")
    
    print("✅ Optimized Spark session created successfully!")
    
    # Monitor performance
    metrics = monitor_spark_performance(spark)
    print(f"📊 Performance metrics: {metrics}")
    
    spark.stop()