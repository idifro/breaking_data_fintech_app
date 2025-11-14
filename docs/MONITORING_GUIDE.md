# Scalability Monitoring Documentation

## 📊 Overview

This comprehensive monitoring system provides real-time scalability metrics, performance tracking, and automated load testing for the ML pipeline. It's designed for the "Data Engineering at Scale" academic project to demonstrate enterprise-level monitoring capabilities without external dashboard dependencies.

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   ML Pipeline   │───▶│ Scalability      │───▶│   MLflow        │
│   (Training/    │    │ Monitor          │    │   Experiments   │
│   Inference)    │    │ (810 lines)      │    │   Tracking      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                               │
                               ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   JSON Reports  │◀───│   Performance    │───▶│   Load Testing  │
│   & Metrics     │    │   Benchmarking   │    │   Framework     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install monitoring dependencies
pip install -r requirements.txt

# Verify core monitoring components
python -c "
from src.scalability_monitor import ScalabilityMonitor
from src.monitoring_config_manager import get_monitoring_config_manager
print('✅ Monitoring components loaded successfully')
"
```

### 2. Start Monitoring Services

```bash
# Run ML pipeline with enhanced monitoring
python scripts/train_model_with_monitoring.py --enable-monitoring

# Run inference with performance tracking
python scripts/inference_with_monitoring.py --enable-monitoring --concurrent-testing

# Execute comprehensive load testing
python scripts/load_testing_scenarios.py --scenario all --duration 300
```

### 3. Access Results

- **MLflow Tracking**: `mlflow ui --host 0.0.0.0 --port 5000` → http://localhost:5000
- **JSON Reports**: `results/evaluation/` directory
- **Log Files**: `logs/pipeline.log`
- **Performance Artifacts**: MLflow experiment artifacts

## 📈 Monitoring Components

### 🔧 Core Monitoring Module (`src/scalability_monitor.py`)

The central monitoring system (810 lines) that provides comprehensive tracking:

- **Data Volume Metrics**: Processing throughput, partition efficiency, data size scaling
- **Performance Metrics**: Response times, processing efficiency, bottleneck detection  
- **Resource Utilization**: CPU, memory, disk I/O monitoring with real-time tracking
- **Scalability Scores**: Linear scalability assessment and efficiency analysis

```python
# Example usage
from src.scalability_monitor import ScalabilityMonitor

monitor = ScalabilityMonitor(config, spark, monitoring_config)
metrics = monitor.monitor_data_volume(spark_df)
performance = monitor.monitor_performance()
resources = monitor.monitor_resource_usage()
scalability_score = monitor.calculate_linear_scalability_score(volume_metrics)
```

### 📊 MLflow Integration (`src/monitoring_config_manager.py`)

Comprehensive MLflow integration for experiment tracking:

- **Experiment Management**: Separate monitoring and training experiments
- **Metrics Logging**: Performance, scalability, and resource utilization metrics
- **Artifact Storage**: Monitoring reports, performance plots, and analysis results
- **Run Tracking**: Automated experiment organization with timestamped runs

**Key Features:**
- Automatic experiment creation and management
- Structured artifact organization
- Historical performance tracking
- Integration with monitoring workflows

### 🏋️ Load Testing Framework (`scripts/load_testing_scenarios.py`)

Comprehensive load testing system (875 lines) with multiple scenarios:

1. **Baseline Testing**: Single-user performance measurement
2. **Concurrency Testing**: Multi-user concurrent load simulation
3. **Volume Testing**: Data size scaling performance analysis
4. **Stress Testing**: System breaking point identification
5. **Regression Testing**: Performance comparison and trend analysis

```bash
# Run comprehensive load testing
python scripts/load_testing_scenarios.py --scenario all --duration 300

# Individual testing scenarios
python scripts/load_testing_scenarios.py --scenario baseline --duration 60
python scripts/load_testing_scenarios.py --scenario stress --duration 180
python scripts/load_testing_scenarios.py --scenario volume --max-volume 1000
```

### 🚀 Enhanced Training (`scripts/train_model_with_monitoring.py`)

Production training pipeline (757 lines) with integrated monitoring:

- **Real-time Performance Tracking**: During model training execution
- **Resource Monitoring**: CPU, memory, and processing efficiency
- **MLflow Integration**: Automatic experiment logging with monitoring metrics
- **Load Testing Integration**: Optional concurrent training load testing

**Usage Examples:**
```bash
# Enhanced training with monitoring
python scripts/train_model_with_monitoring.py --enable-monitoring

# Training with load testing
python scripts/train_model_with_monitoring.py --load-test --concurrent-users 5

# Specific stocks with detailed monitoring
python scripts/train_model_with_monitoring.py --stocks AAPL,GOOG,NVDA --enable-monitoring
```

### ⚡ Enhanced Inference (`scripts/inference_with_monitoring.py`)

Production inference pipeline (1415 lines) with scalability monitoring:

- **Real-time Inference Monitoring**: Performance tracking during predictions
- **Concurrent Testing**: Multi-threaded inference load testing
- **Quality Checks**: Data validation and prediction quality monitoring
- **Throughput Analysis**: Records/second processing with efficiency metrics

**Usage Examples:**
```bash
# Monitored inference with performance tracking
python scripts/inference_with_monitoring.py --enable-monitoring

# Concurrent inference testing
python scripts/inference_with_monitoring.py --concurrent-testing --max-users 10

# Load testing with quality checks
python scripts/inference_with_monitoring.py --load-test --validate-quality
```

## ⚙️ Configuration

### YAML Configuration (`config/config.yaml`)

```yaml
monitoring:
  enabled: true
  detailed_metrics: true
  experiment_name: pipeline_scalability_monitoring
  
  # Performance thresholds
  thresholds:
    min_throughput_records_per_second: 100
    max_memory_usage_percent: 85
    max_cpu_usage_percent: 80
    min_scalability_score: 0.6
    max_processing_time_per_record_ms: 100
    min_partition_efficiency: 0.7
  
  # Resource monitoring settings
  resource_monitoring_interval: 1.0
  cpu_threshold_warning_percent: 85
  memory_threshold_warning_mb: 8192
  
  # Load testing configuration
  load_testing:
    concurrent_users: [1, 5, 10, 20]
    data_volume_multipliers: [1, 2, 5, 10]
    benchmark_duration_minutes: 5
  
  # Output settings
  generate_reports: true
  save_metrics_json: true
  console_output: true
  separate_experiment: true
  
  # Alert settings
  alerts:
    enabled: true
    performance_degradation_threshold: 0.3
    email_notifications: false
    slack_webhook: null

# MLflow Integration
mlflow:
  experiment_name: stock_forecasting_gbt_exp
  log_artifacts: true
  log_metrics: true
  log_models: true
  log_params: true
```

### Environment Variables (`.env`)

```bash
# MLflow settings
MLFLOW_TRACKING_URI=./mlflow_tracking
MLFLOW_EXPERIMENT_NAME=pipeline_scalability_monitoring

# Spark settings
SPARK_LOCAL_DIRS=/tmp/spark
SPARK_DRIVER_MEMORY=4g
SPARK_EXECUTOR_MEMORY=8g
PYSPARK_PYTHON=python

# Monitoring settings
MONITORING_ENABLED=true
MONITORING_LOG_LEVEL=INFO
MONITORING_SEPARATE_EXPERIMENT=true

# Performance settings
ENABLE_DETAILED_METRICS=true
RESOURCE_MONITORING_INTERVAL=1.0
```

## 📊 Metrics Reference

### Core Scalability Metrics

| Metric | Description | Unit | Good Range |
|--------|-------------|------|------------|
| `throughput_records_per_second` | Data processing rate | records/sec | >100 |
| `total_processing_time` | End-to-end processing time | seconds | <300 |
| `linear_scalability_score` | How well system scales linearly | 0.0-1.0 | >0.6 |
| `peak_memory_usage_mb` | Maximum memory consumption | MB | <85% total |
| `avg_cpu_usage_percent` | Average CPU utilization | % | <80% |
| `partition_efficiency_score` | Data partitioning efficiency | 0.0-1.0 | >0.7 |
| `resource_efficiency_score` | Overall resource utilization efficiency | 0.0-1.0 | >0.6 |

### Performance Benchmarking Metrics

| Metric | Description | Unit | Alert Threshold |
|--------|-------------|------|-----------------|
| `feature_engineering_time` | Time for feature processing | seconds | >60s |
| `model_inference_time` | Model prediction time | seconds | >30s |
| `avg_processing_time_per_record` | Per-record processing time | milliseconds | >100ms |
| `disk_io_read_mb` | Disk read operations | MB | Variable |
| `disk_io_write_mb` | Disk write operations | MB | Variable |

### System Health Metrics

| Metric | Description | Unit | Alert Threshold |
|--------|-------------|------|-----------------|
| `cpu_percent` | Current CPU usage | % | >85% |
| `memory_percent` | Current memory usage | % | >85% |
| `disk_percent` | Current disk usage | % | >90% |
| `load_average` | System load average | float | >CPU cores |

### Performance Benchmarks

Based on typical performance expectations for different data volumes:

**Small Dataset (1-10 MB):**
- Throughput: >1000 records/sec
- Processing Time: <30 seconds
- Memory Usage: <1 GB
- Scalability Score: >0.8

**Medium Dataset (10-100 MB):**
- Throughput: >500 records/sec  
- Processing Time: <120 seconds
- Memory Usage: <4 GB
- Scalability Score: >0.7

**Large Dataset (100+ MB):**
- Throughput: >100 records/sec
- Processing Time: <300 seconds
- Memory Usage: <8 GB
- Scalability Score: >0.6

## 🔍 Usage Examples

### Enhanced Training with Monitoring

```python
from scripts.train_model_with_monitoring import MonitoredModelTrainer

# Initialize monitored trainer
trainer = MonitoredModelTrainer(config_path="config/config.yaml")

# Run training with comprehensive monitoring
results = trainer.train_models_with_monitoring(
    stocks=['AAPL', 'GOOG', 'NVDA'],
    enable_monitoring=True,
    run_load_tests=True,
    concurrent_users=5
)

# Access monitoring results
print(f"Processing efficiency: {results['scalability_metrics']['resource_efficiency_score']:.3f}")
print(f"Linear scalability: {results['scalability_metrics']['linear_scalability_score']:.3f}")
print(f"Peak memory usage: {results['scalability_metrics']['peak_memory_usage_mb']:.1f} MB")
```

### Monitored Inference

```python
from scripts.inference_with_monitoring import MonitoredInferenceEngine

# Initialize monitored inference
engine = MonitoredInferenceEngine(config_path="config/config.yaml")

# Run inference with performance tracking
predictions = engine.run_monitored_inference(
    stocks=['AAPL', 'TSLA'],
    enable_monitoring=True,
    concurrent_testing=True,
    max_concurrent_users=10
)

# View performance metrics
performance = predictions['monitoring_results']['performance_metrics']
print(f"Throughput: {performance['throughput_records_per_second']:.1f} records/sec")
print(f"Processing time: {performance['total_processing_time']:.2f}s")
print(f"Scalability score: {performance['linear_scalability_score']:.3f}")
```

### Comprehensive Load Testing

```python
from scripts.load_testing_scenarios import LoadTestRunner

# Initialize load test runner
runner = LoadTestRunner()

# Run individual scenario
baseline_results = runner.run_scenario("baseline", duration_seconds=60)

# Run comprehensive testing suite
all_results = runner.run_all_scenarios(duration_per_scenario=90)

# Analyze performance results
print(f"Maximum throughput: {baseline_results['max_throughput']:.2f} records/sec")
print(f"P95 response time: {baseline_results['response_time_p95']:.2f}s")
print(f"Resource efficiency: {baseline_results['resource_efficiency']:.3f}")

# Performance regression analysis
regression_detected = runner.detect_performance_regression(
    current_results=baseline_results,
    baseline_file="results/baseline_metrics.json"
)
```

### MLflow Integration Examples

```python
from src.monitoring_config_manager import get_monitoring_config_manager

# Access monitoring configuration
cm = get_monitoring_config_manager()

# View recent monitoring experiments
training_config = cm.get_monitoring_config('training_scalability')
print(f"Latest run: {training_config['latest_run_id']}")
print(f"Experiment: {training_config['experiment_name']}")

# Access experiment artifacts
artifacts_url = training_config['artifacts_url']
metrics_url = training_config['metrics_url']
```

## 📈 Interpreting Results

### Scalability Score Interpretation

- **0.9-1.0**: Excellent linear scalability
- **0.7-0.9**: Good scalability with minor inefficiencies
- **0.5-0.7**: Moderate scalability, some bottlenecks present
- **<0.5**: Poor scalability, significant optimization needed

### Throughput Analysis

Expected throughput ranges by data volume:
- **Small files (1-10 MB)**: 10-50 MB/s
- **Medium files (10-100 MB)**: 50-200 MB/s
- **Large files (100+ MB)**: 100-500 MB/s

### Resource Utilization Guidelines

**CPU Usage:**
- <50%: Light load, good headroom
- 50-80%: Normal load, efficient utilization
- 80-95%: Heavy load, monitor for bottlenecks
- >95%: Critical load, likely performance impact

**Memory Usage:**
- <60%: Comfortable usage
- 60-80%: Normal usage
- 80-90%: High usage, monitor closely
- >90%: Critical, risk of OOM errors

## 🚨 Troubleshooting

## 🚨 Troubleshooting

### Common Issues

#### Monitoring Not Starting
```bash
# Check monitoring configuration
python -c "
from src.utils import get_config
config = get_config()
print(f'Monitoring enabled: {config.monitoring.enabled}')
print(f'Thresholds: {config.monitoring.thresholds}')
"

# Verify scalability monitor initialization
python -c "
from src.scalability_monitor import ScalabilityMonitor
from src.utils import get_config, get_spark_session
config = get_config()
spark = get_spark_session(config)
monitor = ScalabilityMonitor(config, spark)
print('✅ Monitoring components initialized successfully')
"
```

#### High Memory Usage
```bash
# Check current memory usage
free -h

# Monitor memory during execution
watch -n 1 'free -h'

# Reduce Spark memory allocation
export SPARK_DRIVER_MEMORY=2g
export SPARK_EXECUTOR_MEMORY=4g

# Run with reduced memory settings
python scripts/train_model_with_monitoring.py --enable-monitoring
```

#### Slow Performance
```bash
# Check CPU utilization
htop

# Monitor I/O performance
iostat -x 1

# Check Spark configuration
python -c "
from src.utils import get_config
config = get_config()
print('Spark configs:', config.spark.configs)
"

# Optimize for performance
export SPARK_SQL_ADAPTIVE_ENABLED=true
export SPARK_SQL_ADAPTIVE_COALESCEPARTITIONS_ENABLED=true
```

#### MLflow Tracking Issues
```bash
# Check MLflow setup
mlflow doctor

# Verify tracking URI
echo $MLFLOW_TRACKING_URI

# Test MLflow connectivity
python -c "
import mlflow
from mlflow.tracking import MlflowClient
client = MlflowClient()
experiments = client.search_experiments()
print(f'Found {len(experiments)} experiments')
"

# Check monitoring experiments
python -c "
from src.monitoring_config_manager import get_monitoring_config_manager
cm = get_monitoring_config_manager()
configs = cm.get_all_monitoring_configs()
print('Available experiments:', list(configs.keys()))
"
```

### Log Analysis

Monitor logs for performance insights:
```bash
# Real-time monitoring logs
tail -f logs/pipeline.log | grep -E "MONITOR|SCALABILITY"

# Performance metrics patterns
grep "throughput\|scalability\|efficiency" logs/pipeline.log | tail -20

# Memory and resource warnings
grep -i -E "memory|cpu|resource" logs/pipeline.log

# Error analysis
grep -E "ERROR|CRITICAL|WARNING" logs/pipeline.log | tail -10
```

### Performance Optimization

#### Spark Configuration Tuning
```python
# Optimal Spark settings for monitoring
spark_optimizations = {
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    "spark.sql.adaptive.skewJoin.enabled": "true",
    "spark.serializer": "org.apache.spark.serializer.KryloSerializer",
    "spark.sql.execution.arrow.pyspark.enabled": "true",
    "spark.sql.adaptive.advisoryPartitionSizeInBytes": "64MB"
}

# Apply via environment or config.yaml
```

#### Memory Management
```python
# Monitor memory usage in monitoring code
import psutil
from src.scalability_monitor import ScalabilityMonitor

def check_memory_efficiency():
    memory = psutil.virtual_memory()
    print(f"Memory usage: {memory.percent:.1f}%")
    
    if memory.percent > 85:
        print("Warning: High memory usage detected")
        print("Consider reducing data volume or increasing memory allocation")
```

#### Resource Monitoring Optimization
```bash
# Reduce monitoring overhead
export MONITORING_RESOURCE_INTERVAL=2.0  # Increase interval
export MONITORING_DETAILED_METRICS=false  # Disable detailed metrics

# Run with optimized monitoring
python scripts/train_model_with_monitoring.py --enable-monitoring
```

## 🎯 Best Practices

### Development Workflow

1. **Start with Baseline Metrics**: Always establish performance baselines before optimization
2. **Incremental Load Testing**: Test with gradually increasing data volumes and concurrency
3. **Monitor Continuously**: Keep monitoring enabled during development iterations
4. **Document Performance Changes**: Record configuration changes and their performance impact
5. **Regular Regression Testing**: Run weekly load testing to detect performance degradation
6. **Resource Planning**: Size infrastructure based on monitoring results and load testing

### Production Deployment

1. **Threshold Configuration**: Adjust monitoring thresholds based on production requirements
2. **Automated Monitoring**: Enable monitoring in all production pipelines
3. **Alert Management**: Configure appropriate alert thresholds for production workloads
4. **Performance Reporting**: Generate regular monitoring reports for stakeholder review
5. **Capacity Planning**: Use scalability metrics for infrastructure sizing decisions
6. **Historical Analysis**: Maintain monitoring history for trend analysis

### Monitoring Data Management

1. **MLflow Experiment Organization**: Separate monitoring and training experiments
2. **Artifact Retention**: Keep monitoring artifacts for historical analysis
3. **Data Archiving**: Archive old monitoring data while maintaining performance summaries
4. **Report Generation**: Automated generation of monitoring reports and analysis
5. **Backup Strategy**: Regular backup of MLflow tracking data and monitoring results

## 📚 Additional Resources

### Documentation References
- [Apache Spark Performance Tuning](https://spark.apache.org/docs/latest/sql-performance-tuning.html)
- [MLflow Tracking Documentation](https://mlflow.org/docs/latest/tracking.html)
- [Delta Lake Performance Guide](https://docs.delta.io/latest/optimizations-oss.html)
- [PySpark Monitoring Guide](https://spark.apache.org/docs/latest/monitoring.html)

### Monitoring Best Practices
- [Observability Patterns](https://martinfowler.com/articles/domain-oriented-observability.html)
- [Performance Monitoring Guide](https://sre.google/sre-book/monitoring-distributed-systems/)
- [Scalability Testing Patterns](https://martinfowler.com/articles/practical-test-pyramid.html)

### Load Testing Resources
- [Performance Testing Strategies](https://martinfowler.com/articles/load-testing.html)
- [Scalability Testing Guide](https://www.perfmatrix.com/scalability-testing/)

## 🤝 Contributing

This monitoring system can be extended with additional capabilities:

1. **Custom Metrics**: Add domain-specific metrics in `ScalabilityMonitor`
2. **New Scenarios**: Create specialized load testing scenarios 
3. **Enhanced Reporting**: Develop additional analysis and reporting features
4. **Integration Points**: Add monitoring hooks in custom pipeline components
5. **Visualization Enhancement**: Create additional monitoring plots and analysis

## 📞 Support

For issues with the monitoring system:

1. **Check Configuration**: Verify `config/config.yaml` monitoring settings
2. **Review Logs**: Examine `logs/pipeline.log` for error messages
3. **Test Components**: Verify individual monitoring components separately
4. **MLflow Verification**: Ensure MLflow tracking is working correctly
5. **Resource Monitoring**: Check system resources and Spark configuration

This monitoring system provides comprehensive observability for ML pipeline scalability, enabling data-driven optimization and performance analysis for production Data Engineering at Scale projects.