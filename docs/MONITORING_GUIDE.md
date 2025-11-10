# Scalability Monitoring Documentation

## 📊 Overview

This comprehensive monitoring system provides real-time scalability metrics, performance tracking, and automated load testing for the ML pipeline. It's designed for the "Data Engineering at Scale" academic project to demonstrate enterprise-level monitoring capabilities.

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   ML Pipeline   │───▶│ Scalability      │───▶│   MLflow        │
│   (Training/    │    │ Monitor          │    │   Experiments   │
│   Inference)    │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                               │
                               ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit     │◀───│   FastAPI        │───▶│   Load Testing  │
│   Dashboard     │    │   API Service    │    │   Scenarios     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install monitoring dependencies
pip install -r requirements.txt

# Or install specific monitoring packages
pip install streamlit fastapi uvicorn psutil plotly pydantic websockets
```

### 2. Start Monitoring Services

```bash
# Terminal 1: Start FastAPI monitoring service
uvicorn dashboard.monitoring_api:app --host 0.0.0.0 --port 8502 --reload

# Terminal 2: Start Streamlit dashboard
streamlit run dashboard/monitoring_dashboard.py --server.port 8501

# Terminal 3: Run ML pipeline with monitoring
python scripts/train_model_with_monitoring.py --enable-monitoring --concurrent-load-test
```

### 3. Access Interfaces

- **Streamlit Dashboard**: http://localhost:8501
- **FastAPI Service**: http://localhost:8502
- **API Documentation**: http://localhost:8502/docs

## 📈 Monitoring Components

### 🔧 Core Monitoring Module (`src/scalability_monitor.py`)

The central monitoring system that tracks:

- **Data Volume Metrics**: Processing throughput, data size scaling
- **Performance Metrics**: Response times, processing efficiency
- **Resource Utilization**: CPU, memory, disk usage
- **Scalability Scores**: Linear scalability assessment

```python
# Example usage
from src.scalability_monitor import ScalabilityMonitor

monitor = ScalabilityMonitor(config)
metrics = monitor.monitor_data_volume(spark_df)
performance = monitor.monitor_performance()
resources = monitor.monitor_resource_usage()
```

### 📊 Streamlit Dashboard (`dashboard/monitoring_dashboard.py`)

Real-time visualization dashboard featuring:

- **Overview Metrics Cards**: Key performance indicators
- **Alert System**: Threshold violation notifications
- **Performance Trends**: Interactive time-series charts
- **Resource Utilization**: Real-time system monitoring
- **Experiment Comparison**: Side-by-side run analysis

**Key Features:**
- Auto-refresh every 5 seconds (configurable)
- Historical data visualization
- Interactive filtering and controls
- Responsive design for various screen sizes

### 🌐 FastAPI Service (`dashboard/monitoring_api.py`)

RESTful API service providing:

- **Current Metrics**: `/api/metrics/current`
- **Historical Data**: `/api/metrics/history`
- **System Health**: `/api/system/health`
- **Alerts**: `/api/alerts`
- **WebSocket Streaming**: `/ws/metrics`

**API Examples:**
```bash
# Get current metrics
curl http://localhost:8502/api/metrics/current

# Get historical data
curl "http://localhost:8502/api/metrics/history?hours=24&limit=100"

# Check system health
curl http://localhost:8502/api/system/health
```

### 🏋️ Load Testing (`scripts/load_testing_scenarios.py`)

Comprehensive load testing with multiple scenarios:

1. **Baseline**: Single-user performance measurement
2. **Concurrency**: Multi-user concurrent testing
3. **Volume**: Data size scaling tests
4. **Stress**: System limit identification
5. **Regression**: Performance comparison analysis

```bash
# Run individual scenarios
python scripts/load_testing_scenarios.py --scenario baseline --duration 60
python scripts/load_testing_scenarios.py --scenario concurrency --max-users 20
python scripts/load_testing_scenarios.py --scenario stress --duration 300

# Run comprehensive test suite
python scripts/load_testing_scenarios.py --scenario all --duration 120
```

## ⚙️ Configuration

### YAML Configuration (`config/config.yaml`)

```yaml
monitoring:
  enabled: true
  
  # Performance thresholds
  thresholds:
    min_throughput_records_per_second: 1000
    max_memory_usage_percent: 80
    max_cpu_usage_percent: 85
    min_scalability_score: 0.7
  
  # Dashboard settings
  dashboard:
    port: 8501
    auto_refresh_seconds: 5
    historical_data_points: 100
  
  # API settings
  api:
    port: 8502
    enable_cors: true
  
  # Load testing parameters
  load_testing:
    max_concurrent_users: 20
    stress_test_duration_minutes: 5
    data_volume_sizes_mb: [1, 5, 10, 20, 50, 100]
```

### Environment Variables (`.env`)

```bash
# MLflow settings
MLFLOW_TRACKING_URI=./mlflow_tracking
MLFLOW_EXPERIMENT_NAME=pipeline_scalability_monitoring

# Spark settings
SPARK_LOCAL_DIRS=/tmp/spark
PYSPARK_PYTHON=python

# Monitoring settings
MONITORING_ENABLED=true
MONITORING_LOG_LEVEL=INFO
```

## 📊 Metrics Reference

### Core Scalability Metrics

| Metric | Description | Unit | Good Range |
|--------|-------------|------|------------|
| `throughput_records_per_second` | Data processing rate | records/sec | >1000 |
| `total_processing_time` | End-to-end processing time | seconds | <60 |
| `linear_scalability_score` | How well system scales linearly | 0.0-1.0 | >0.7 |
| `peak_memory_usage_mb` | Maximum memory consumption | MB | <80% total |
| `avg_cpu_usage_percent` | Average CPU utilization | % | <85% |
| `data_volume_gb` | Amount of data processed | GB | Variable |

### System Health Metrics

| Metric | Description | Unit | Alert Threshold |
|--------|-------------|------|-----------------|
| `cpu_percent` | Current CPU usage | % | >90% |
| `memory_percent` | Current memory usage | % | >85% |
| `disk_percent` | Current disk usage | % | >90% |
| `load_average` | System load average | float | >CPU cores |

### Performance Benchmarks

Based on typical performance expectations:

**Small Dataset (1-10 MB):**
- Throughput: >5000 records/sec
- Processing Time: <10 seconds
- Memory Usage: <500 MB

**Medium Dataset (10-100 MB):**
- Throughput: >2000 records/sec
- Processing Time: <60 seconds
- Memory Usage: <2 GB

**Large Dataset (100+ MB):**
- Throughput: >1000 records/sec
- Processing Time: <300 seconds
- Memory Usage: <8 GB

## 🔍 Usage Examples

### Enhanced Training with Monitoring

```python
from scripts.train_model_with_monitoring import MonitoredModelTrainer

# Initialize monitored trainer
trainer = MonitoredModelTrainer(config_path="config/config.yaml")

# Run training with monitoring
results = trainer.train_models_with_monitoring(
    enable_monitoring=True,
    run_load_tests=True,
    concurrent_users=5
)

# Access monitoring metrics
print(f"Training efficiency: {results['efficiency_metrics']['processing_efficiency']:.2f}")
print(f"Scalability score: {results['scalability_metrics']['linear_scalability_score']:.3f}")
```

### Monitored Inference

```python
from scripts.inference_with_monitoring import MonitoredInferenceEngine

# Initialize monitored inference
engine = MonitoredInferenceEngine(config_path="config/config.yaml")

# Run inference with monitoring
predictions = engine.run_monitored_inference(
    enable_monitoring=True,
    concurrent_testing=True,
    max_concurrent_users=10
)

# View performance metrics
print(f"Inference throughput: {predictions['performance']['throughput']:.1f} predictions/sec")
```

### Load Testing Integration

```python
from scripts.load_testing_scenarios import LoadTestRunner

# Run comprehensive load tests
runner = LoadTestRunner()

# Individual scenario
baseline_results = await runner.run_scenario("baseline", duration_seconds=60)

# Full test suite
all_results = await runner.run_all_scenarios(duration_per_scenario=90)

# Analyze results
print(f"Maximum throughput: {baseline_results['throughput']['max']:.2f} MB/s")
print(f"P95 response time: {baseline_results['response_time']['p95']:.2f}s")
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

### Common Issues

#### High Memory Usage
```bash
# Check current memory usage
free -h

# Monitor memory during execution
watch -n 1 'free -h'

# Reduce Spark memory allocation
export SPARK_DRIVER_MEMORY=4g
export SPARK_EXECUTOR_MEMORY=2g
```

#### Slow Performance
```bash
# Check CPU utilization
htop

# Monitor disk I/O
iostat -x 1

# Check for competing processes
ps aux --sort=-%cpu | head -10
```

#### Dashboard Not Loading
```bash
# Check if services are running
netstat -tulpn | grep -E '8501|8502'

# Restart dashboard
streamlit run dashboard/monitoring_dashboard.py --server.port 8501

# Check logs
tail -f logs/monitoring.log
```

### Log Analysis

Monitor logs for performance insights:
```bash
# Real-time monitoring
tail -f logs/monitoring.log | grep -E "WARNING|ERROR"

# Performance patterns
grep "throughput" logs/monitoring.log | tail -20

# Memory warnings
grep -i "memory" logs/monitoring.log
```

### Performance Optimization

#### Spark Tuning
```python
# Optimize Spark configuration
spark_config = {
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    "spark.sql.adaptive.skewJoin.enabled": "true",
    "spark.serializer": "org.apache.spark.serializer.KryoSerializer"
}
```

#### Memory Management
```python
# Monitor memory usage in code
import psutil

def monitor_memory():
    memory = psutil.virtual_memory()
    print(f"Memory usage: {memory.percent:.1f}%")
    
    if memory.percent > 85:
        print("Warning: High memory usage detected")
```

## 🎯 Best Practices

### Development Workflow

1. **Start with Baseline**: Always establish baseline metrics
2. **Incremental Testing**: Test with gradually increasing loads
3. **Monitor Continuously**: Keep dashboard running during development
4. **Document Changes**: Record configuration changes and their impact
5. **Regular Load Testing**: Run weekly performance regression tests

### Production Deployment

1. **Threshold Tuning**: Adjust alert thresholds based on production data
2. **Resource Planning**: Size infrastructure based on load test results
3. **Monitoring Alerts**: Set up automated alerting for threshold violations
4. **Regular Reporting**: Generate weekly performance reports
5. **Capacity Planning**: Use trend data for infrastructure scaling

### Data Management

1. **Historical Retention**: Keep 30+ days of monitoring data
2. **Backup Strategy**: Regular backup of MLflow experiments
3. **Data Archiving**: Archive old test results but maintain summaries
4. **Cleanup Policies**: Automated cleanup of temporary monitoring files

## 📚 Additional Resources

### Documentation
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PySpark Performance Tuning](https://spark.apache.org/docs/latest/sql-performance-tuning.html)

### Monitoring Tools
- [Grafana](https://grafana.com/) - Advanced dashboarding
- [Prometheus](https://prometheus.io/) - Metrics collection
- [Jaeger](https://www.jaegertracing.io/) - Distributed tracing
- [New Relic](https://newrelic.com/) - APM monitoring

### Load Testing Tools
- [Locust](https://locust.io/) - Python-based load testing
- [Artillery](https://artillery.io/) - Modern load testing toolkit
- [JMeter](https://jmeter.apache.org/) - Traditional load testing
- [K6](https://k6.io/) - Developer-centric load testing

## 🤝 Contributing

This monitoring system is designed for academic demonstration but can be extended:

1. **New Metrics**: Add custom metrics in `ScalabilityMonitor`
2. **Dashboard Widgets**: Create new Streamlit components
3. **API Endpoints**: Extend FastAPI service with new routes
4. **Load Test Scenarios**: Add specialized testing scenarios
5. **Visualization**: Enhance charts and graphs with Plotly

## 📞 Support

For issues or questions about the monitoring system:

1. Check logs in `logs/monitoring.log`
2. Review configuration in `config/config.yaml`
3. Verify dependencies in `requirements.txt`
4. Test individual components separately
5. Consult troubleshooting section above

This monitoring system provides comprehensive observability for ML pipeline scalability, enabling data-driven optimization and performance analysis for academic and production use cases.