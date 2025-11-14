# Scalability Monitoring Quick Reference

## 🚀 Quick Start Commands

### Installation & Setup
```bash
# Ensure you're in the correct conda environment
conda activate breaking_data

# Navigate to pipeline directory
cd /home/mha2cob/Downloads/breaking_data/spark_ml_pipeline

# Install dependencies
pip install -r requirements.txt

# Verify monitoring components
python -c "
from src.scalability_monitor import ScalabilityMonitor
from src.monitoring_config_manager import get_monitoring_config_manager
print('✅ Monitoring system ready')
"
```

### Enhanced Training with Monitoring
```bash
# Basic training with monitoring
python scripts/train_model_with_monitoring.py --enable-monitoring

# Train specific stocks with detailed monitoring
python scripts/train_model_with_monitoring.py --stocks AAPL,GOOG,NVDA --enable-monitoring

# Training with load testing
python scripts/train_model_with_monitoring.py --load-test --concurrent-users 5

# Disable monitoring for faster training
python scripts/train_model_with_monitoring.py --no-monitoring
```

### Enhanced Inference with Monitoring
```bash
# Basic inference with monitoring
python scripts/inference_with_monitoring.py --enable-monitoring

# Inference with concurrent testing
python scripts/inference_with_monitoring.py --concurrent-testing --max-users 10

# Inference with quality checks
python scripts/inference_with_monitoring.py --load-test --validate-quality

# Inference for specific stocks
python scripts/inference_with_monitoring.py --stocks AAPL,TSLA --enable-monitoring
```

### Load Testing Scenarios
```bash
# Comprehensive load testing (all scenarios)
python scripts/load_testing_scenarios.py --scenario all --duration 300

# Individual scenarios
python scripts/load_testing_scenarios.py --scenario baseline --duration 60
python scripts/load_testing_scenarios.py --scenario stress --duration 180
python scripts/load_testing_scenarios.py --scenario volume --max-volume 1000
python scripts/load_testing_scenarios.py --scenario concurrency --max-users 20
python scripts/load_testing_scenarios.py --scenario regression --duration 120
```

## 📊 Access Points & Results

| Service | URL/Location | Purpose |
|---------|--------------|---------|
| **MLflow UI** | `mlflow ui --host 0.0.0.0 --port 5000` | Experiment tracking & artifacts |
| **Monitoring Logs** | `logs/pipeline.log` | Real-time monitoring output |
| **JSON Reports** | `results/evaluation/` | Performance analysis reports |
| **Monitoring Artifacts** | MLflow experiments | Stored monitoring data |
| **Performance Plots** | `plots/` | Generated visualization files |

### Quick Access Commands
```bash
# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# View real-time logs
tail -f logs/pipeline.log

# Check monitoring results
find results/ -name "*scalability*" -type f

# View recent MLflow experiments
python -c "
from src.monitoring_config_manager import get_monitoring_config_manager
cm = get_monitoring_config_manager()
configs = cm.get_all_monitoring_configs()
for name, config in configs.items():
    print(f'{name}: {config.get(\"latest_run_id\", \"No runs\")[:8]}...')
"
```
## 🔧 Configuration Quick Settings

### Enable/Disable Monitoring
```yaml
# config/config.yaml
monitoring:
  enabled: true
  detailed_metrics: true
  generate_reports: true
  save_metrics_json: true
```

### Adjust Performance Thresholds
```yaml
monitoring:
  thresholds:
    min_throughput_records_per_second: 100
    max_memory_usage_percent: 85
    max_cpu_usage_percent: 80
    min_scalability_score: 0.6
    max_processing_time_per_record_ms: 100
    min_partition_efficiency: 0.7
```

### Load Testing Configuration
```yaml
monitoring:
  load_testing:
    concurrent_users: [1, 5, 10, 20]
    data_volume_multipliers: [1, 2, 5, 10]
    benchmark_duration_minutes: 5
```

### Environment Variables
```bash
# Set monitoring preferences
export MONITORING_ENABLED=true
export MONITORING_DETAILED_METRICS=true
export MONITORING_RESOURCE_INTERVAL=1.0

# Spark optimization settings
export SPARK_DRIVER_MEMORY=4g
export SPARK_EXECUTOR_MEMORY=8g
export SPARK_SQL_ADAPTIVE_ENABLED=true
```

## 📈 Key Metrics & Thresholds

| Metric | Good Value | Alert Threshold | Unit | Description |
|--------|------------|-----------------|------|-------------|
| **Throughput** | >1000 | <100 | records/sec | Data processing rate |
| **Scalability Score** | >0.7 | <0.6 | 0.0-1.0 | Linear scaling efficiency |
| **Resource Efficiency** | >0.6 | <0.5 | 0.0-1.0 | Overall resource utilization |
| **Memory Usage** | <85% | >85% | % | Peak memory consumption |
| **CPU Usage** | <80% | >85% | % | Average CPU utilization |
| **Processing Time** | <120s | >300s | seconds | End-to-end processing |
| **Partition Efficiency** | >0.7 | <0.5 | 0.0-1.0 | Data partitioning quality |

### Performance Ranges by Data Volume

| Data Volume | Expected Throughput | Expected Memory | Expected Time |
|-------------|-------------------|-----------------|---------------|
| **Small (1-10 MB)** | 1000-5000 rec/sec | 500-1000 MB | 10-30s |
| **Medium (10-100 MB)** | 500-2000 rec/sec | 1-4 GB | 30-120s |
| **Large (100+ MB)** | 100-1000 rec/sec | 4-8 GB | 120-300s |

## 🏋️ Load Testing Scenarios

```bash
# Baseline performance
python scripts/load_testing_scenarios.py --scenario baseline --duration 60

# Concurrency testing
python scripts/load_testing_scenarios.py --scenario concurrency --max-users 20

# Volume testing
python scripts/load_testing_scenarios.py --scenario volume --duration 180

# Stress testing
python scripts/load_testing_scenarios.py --scenario stress --duration 300

# Regression detection
python scripts/load_testing_scenarios.py --scenario regression --duration 120

# All scenarios
python scripts/load_testing_scenarios.py --scenario all --duration 120
```

## 🚨 Common Troubleshooting

### Monitoring Issues
```bash
# Check monitoring system status
python -c "
from src.utils import get_config
config = get_config()
print(f'Monitoring enabled: {config.monitoring.enabled}')
print(f'Detailed metrics: {config.monitoring.detailed_metrics}')
"

# Verify scalability monitor
python -c "
from src.scalability_monitor import ScalabilityMonitor
from src.utils import get_config, get_spark_session
config = get_config()
spark = get_spark_session(config)
monitor = ScalabilityMonitor(config, spark)
print('✅ Monitoring initialized successfully')
"
```

### High Memory Usage
```bash
# Check current memory
free -h

# Reduce Spark memory allocation
export SPARK_DRIVER_MEMORY=2g
export SPARK_EXECUTOR_MEMORY=4g

# Monitor memory during execution
watch -n 1 'free -h'
```

### Slow Performance
```bash
# Check system resources
htop
iostat -x 1

# Optimize Spark settings
export SPARK_SQL_ADAPTIVE_ENABLED=true
export SPARK_SQL_ADAPTIVE_COALESCEPARTITIONS_ENABLED=true

# Check Spark UI (if running)
# http://localhost:4040
```

### MLflow Issues
```bash
# Verify MLflow setup
mlflow doctor

# Check tracking URI
echo $MLFLOW_TRACKING_URI

# Test MLflow connectivity
python -c "
import mlflow
from mlflow.tracking import MlflowClient
client = MlflowClient()
print(f'MLflow connection: OK')
"

# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000
```

### Load Testing Problems
```bash
# Check available scenarios
python scripts/load_testing_scenarios.py --help

# Run quick baseline test
python scripts/load_testing_scenarios.py --scenario baseline --duration 30

# Check logs for errors
tail -f logs/pipeline.log | grep -E "ERROR|CRITICAL"
```

## 📁 Directory Structure & Key Files

```
spark_ml_pipeline/
├── src/                          
│   ├── scalability_monitor.py   # Core monitoring (810 lines)
│   ├── monitoring_config_manager.py  # MLflow integration
│   ├── utils.py                 # Configuration management
│   ├── data_processing.py       # Delta Lake processing
│   ├── feature_engineering.py  # Feature creation (336 lines)
│   ├── model_training.py        # GBT training
│   └── visualization.py         # Plotting and analysis
├── scripts/                      
│   ├── train_model_with_monitoring.py    # Enhanced training (757 lines)
│   ├── inference_with_monitoring.py      # Enhanced inference (1415 lines)
│   ├── load_testing_scenarios.py         # Load testing framework (875 lines)
│   ├── train_model.py          # Basic training
│   └── inference.py            # Basic inference
├── config/
│   └── config.yaml             # Centralized configuration
├── docs/
│   ├── MONITORING_GUIDE.md     # Comprehensive monitoring guide
│   ├── PERFORMANCE_BENCHMARKS.md  # Performance analysis
│   └── QUICK_REFERENCE.md      # This file
├── results/
│   ├── evaluation/             # Performance reports & JSON metrics
│   ├── training/               # Training results
│   └── inference/              # Inference results
├── mlflow_tracking/            # MLflow experiments & artifacts
├── logs/                       # Application & monitoring logs
└── requirements.txt            # Dependencies
```

## 📊 Monitoring Results Access

View monitoring results in:
- MLflow UI: `mlflow ui --host 0.0.0.0 --port 5000`
- Log files: `logs/pipeline.log`
- JSON reports: `results/evaluation/`

## 💡 Performance Tips & Best Practices

### For Better Throughput
```bash
# Enable Spark optimizations
export SPARK_SQL_ADAPTIVE_ENABLED=true
export SPARK_SQL_ADAPTIVE_COALESCEPARTITIONS_ENABLED=true
export SPARK_SQL_ADAPTIVE_SKEWJOIN_ENABLED=true

# Optimize serialization
export SPARK_SERIALIZER="org.apache.spark.serializer.KryloSerializer"

# Increase partitions for large datasets
# Edit config.yaml: spark.sql.shuffle.partitions: 400
```

### For Lower Memory Usage
```bash
# Reduce driver/executor memory
export SPARK_DRIVER_MEMORY=2g
export SPARK_EXECUTOR_MEMORY=4g

# Enable memory optimization
export SPARK_SQL_EXECUTION_ARROW_PYSPARK_ENABLED=true

# Monitor memory usage
python -c "
import psutil
mem = psutil.virtual_memory()
print(f'Memory usage: {mem.percent:.1f}%')
"
```

### For Monitoring Accuracy
```bash
# Allow warm-up time
python scripts/train_model_with_monitoring.py --enable-monitoring --warm-up

# Run baseline tests first
python scripts/load_testing_scenarios.py --scenario baseline --duration 60

# Monitor during steady state (not startup)
# Wait 2-3 iterations before measuring performance

# Account for JVM garbage collection in measurements
```

### For Better Load Testing
```bash
# Start with small loads
python scripts/load_testing_scenarios.py --scenario baseline --duration 30

# Gradually increase load
python scripts/load_testing_scenarios.py --scenario volume --max-volume 5

# Run regression testing regularly
python scripts/load_testing_scenarios.py --scenario regression --baseline results/baseline_metrics.json

# Monitor system resources during testing
watch -n 1 'free -h && echo "---" && ps aux --sort=-%cpu | head -5'
```

## 📊 Example Complete Monitoring Session

```bash
# 1. Setup and verification
conda activate breaking_data
cd /home/mha2cob/Downloads/breaking_data/spark_ml_pipeline

# Verify system is ready
python -c "
from src.scalability_monitor import ScalabilityMonitor
from src.monitoring_config_manager import get_monitoring_config_manager
print('✅ All monitoring components verified')
"

# 2. Run baseline performance test
echo "🚀 Running baseline performance test..."
python scripts/load_testing_scenarios.py --scenario baseline --duration 60

# 3. Train model with comprehensive monitoring
echo "🚀 Training model with monitoring..."
python scripts/train_model_with_monitoring.py --enable-monitoring --stocks AAPL,GOOG

# 4. Run inference with performance tracking
echo "🚀 Running monitored inference..."
python scripts/inference_with_monitoring.py --enable-monitoring --concurrent-testing

# 5. Start MLflow UI to view results
echo "🚀 Starting MLflow UI..."
mlflow ui --host 0.0.0.0 --port 5000 &

# 6. Run comprehensive load testing
echo "🚀 Running comprehensive load tests..."
python scripts/load_testing_scenarios.py --scenario all --duration 300

# 7. Review results
echo "📊 Checking monitoring results..."

# View recent logs
tail -n 50 logs/pipeline.log

# Check MLflow experiments
python -c "
from src.monitoring_config_manager import get_monitoring_config_manager
cm = get_monitoring_config_manager()
configs = cm.get_all_monitoring_configs()
print('📊 Available monitoring experiments:')
for name, config in configs.items():
    print(f'  - {name}: Run {config.get(\"latest_run_id\", \"N/A\")[:8]}...')
"

# Find performance reports
find results/ -name "*scalability*" -type f -exec echo "📈 Found: {}" \;

echo "✅ Monitoring session complete!"
echo "🌐 Access MLflow UI at: http://localhost:5000"
```

## 🔧 Configuration Examples

### Basic Monitoring Configuration
```yaml
# config/config.yaml
monitoring:
  enabled: true
  detailed_metrics: true
  
mlflow:
  experiment_name: stock_forecasting_gbt_exp
  log_artifacts: true
```

### Advanced Performance Configuration
```yaml
monitoring:
  enabled: true
  detailed_metrics: true
  resource_monitoring_interval: 1.0
  generate_reports: true
  
  thresholds:
    min_throughput_records_per_second: 500
    max_memory_usage_percent: 80
    max_cpu_usage_percent: 85
    min_scalability_score: 0.7
    
  load_testing:
    concurrent_users: [1, 5, 10, 20]
    data_volume_multipliers: [1, 2, 5, 10]

spark:
  driver_memory: 4g
  executor_memory: 8g
  configs:
    spark.sql.adaptive.enabled: "true"
    spark.sql.adaptive.coalescePartitions.enabled: "true"
    spark.serializer: "org.apache.spark.serializer.KryloSerializer"
```

## 📚 Documentation Quick Links

- **📊 MONITORING_GUIDE.md**: Complete monitoring system documentation
- **🚀 PERFORMANCE_BENCHMARKS.md**: Performance analysis and optimization guide
- **⚡ QUICK_REFERENCE.md**: This file - quick command reference
- **📋 README.md**: Complete project documentation with architecture

## 🆘 Emergency Troubleshooting

### System Won't Start
```bash
# Check environment
conda list | grep -E "pyspark|delta|mlflow"

# Verify paths
python -c "import sys; print('\n'.join(sys.path))"

# Reset if needed
conda activate breaking_data
pip install -r requirements.txt --force-reinstall
```

### Performance Issues
```bash
# Immediate memory check
free -h && ps aux --sort=-%mem | head -5

# Quick CPU check  
top -bn1 | grep "Cpu(s)"

# Emergency memory reduction
export SPARK_DRIVER_MEMORY=1g
export SPARK_EXECUTOR_MEMORY=2g
```

### MLflow Issues
```bash
# Quick MLflow reset
rm -rf mlruns/
mlflow doctor
```

This quick reference provides essential commands and troubleshooting steps for effective monitoring system usage.