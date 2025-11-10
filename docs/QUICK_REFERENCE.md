# Scalability Monitoring Quick Reference

## 🚀 Quick Start Commands

### Installation
```bash
pip install -r requirements.txt
```

### Run Monitoring
```bash
# Training with monitoring
python scripts/train_model_with_monitoring.py --enable-monitoring

# Inference with monitoring
python scripts/inference_with_monitoring.py --enable-monitoring --concurrent-testing

# Load testing
python scripts/load_testing_scenarios.py --scenario all --duration 120
```

## 📊 Access Points

| Service | URL | Purpose |
## 🔧 Configuration Quick Settings

### Enable/Disable Monitoring
```yaml
# config/config.yaml
monitoring:
  enabled: true
```

### Adjust Thresholds
```yaml
monitoring:
  thresholds:
    min_throughput_records_per_second: 1000
    max_memory_usage_percent: 80
    max_cpu_usage_percent: 85
    min_scalability_score: 0.7
```

## 📈 Key Metrics

| Metric | Good Value | Alert Threshold | Unit |
|--------|------------|-----------------|------|
| Throughput | >1000 | <500 | records/sec |
| Memory Usage | <80% | >90% | % |
| CPU Usage | <85% | >95% | % |
| Scalability Score | >0.7 | <0.5 | 0.0-1.0 |
| Response Time | <60s | >120s | seconds |

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

### Services Won't Start
```bash
# Check if ports are in use
netstat -tulpn | grep -E '8501|8502'

# Kill existing processes
pkill -f streamlit
pkill -f uvicorn
```

### High Memory Usage
```bash
# Check memory
free -h

# Reduce Spark memory
export SPARK_DRIVER_MEMORY=2g
export SPARK_EXECUTOR_MEMORY=1g
```

## 📁 Directory Structure

```
spark_ml_pipeline/
├── scripts/
│   ├── train_model_with_monitoring.py
│   ├── inference_with_monitoring.py
│   └── load_testing_scenarios.py
├── src/
│   └── scalability_monitor.py   # Core monitoring module
├── config/
│   └── config.yaml              # Configuration
├── docs/
│   ├── MONITORING_GUIDE.md      # Comprehensive guide
│   └── QUICK_REFERENCE.md       # This file
└── requirements.txt             # Dependencies
```

## 📊 Monitoring Results Access

View monitoring results in:
- MLflow UI: `mlflow ui --host 0.0.0.0 --port 5000`
- Log files: `logs/pipeline.log`
- JSON reports: `results/evaluation/`

## 💡 Performance Tips

### For Better Throughput
- Increase Spark partitions: `spark.sql.shuffle.partitions=200`
- Use Kryo serialization: `spark.serializer=KryoSerializer`
- Enable adaptive query execution: `spark.sql.adaptive.enabled=true`

### For Lower Memory Usage
- Reduce driver memory: `SPARK_DRIVER_MEMORY=2g`
- Cache selectively: Use `.cache()` only on reused DataFrames
- Checkpoint long lineages: Use `.checkpoint()` for complex operations

### For Monitoring Accuracy
- Run baseline tests first
- Allow warm-up time (2-3 iterations)
- Monitor during steady state
- Account for JVM garbage collection

## 📊 Example Monitoring Session

```bash
# 1. Start services
# 2. Run baseline test
python scripts/load_testing_scenarios.py --scenario baseline --duration 60

# 3. Run training with monitoring
python scripts/train_model_with_monitoring.py --enable-monitoring

# 4. Check results in MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# 5. Run comprehensive load tests
python scripts/load_testing_scenarios.py --scenario all --duration 300

# 6. Review logs and reports
tail -f logs/pipeline.log
```

This quick reference provides essential commands and settings for effective use of the scalability monitoring system.