#!/usr/bin/env python3
"""
Load Testing Scenarios for ML Pipeline Scalability Monitoring
Automated benchmark scripts for varying data volumes, concurrency levels,
and stress testing scenarios

CONDA ENVIRONMENT: breaking_data
Usage:
    conda activate breaking_data
    python scripts/load_testing_scenarios.py --scenario baseline
    python scripts/load_testing_scenarios.py --scenario stress --duration 300

Features:
- Automated benchmark scenarios
- Varying data volumes and concurrency levels
- Performance regression detection
- Stress testing capabilities
- Comprehensive reporting
- Integration with monitoring pipeline
"""

import argparse
import asyncio
import concurrent.futures
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import threading
import multiprocessing
import psutil
import pandas as pd
import numpy as np

# Add src to path
pipeline_root = Path(__file__).parent.parent
sys.path.insert(0, str(pipeline_root))

from src.utils import Config, Logger, load_environment
from src.scalability_monitor import ScalabilityMonitor, ScalabilityMetrics


class LoadTestingScenario:
    """Base class for load testing scenarios"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.results: List[Dict] = []
        
    async def run(self, duration_seconds: int = 60, **kwargs) -> Dict:
        """Run the load testing scenario"""
        raise NotImplementedError("Subclasses must implement run method")
    
    def analyze_results(self) -> Dict:
        """Analyze results from the scenario"""
        if not self.results:
            return {"error": "No results to analyze"}
        
        # Calculate basic statistics
        throughputs = [r.get('throughput', 0) for r in self.results]
        response_times = [r.get('response_time', 0) for r in self.results]
        memory_usage = [r.get('memory_mb', 0) for r in self.results]
        cpu_usage = [r.get('cpu_percent', 0) for r in self.results]
        
        analysis = {
            "scenario": self.name,
            "total_runs": len(self.results),
            "throughput": {
                "mean": np.mean(throughputs),
                "median": np.median(throughputs),
                "std": np.std(throughputs),
                "min": np.min(throughputs),
                "max": np.max(throughputs),
                "p95": np.percentile(throughputs, 95),
                "p99": np.percentile(throughputs, 99)
            },
            "response_time": {
                "mean": np.mean(response_times),
                "median": np.median(response_times),
                "std": np.std(response_times),
                "min": np.min(response_times),
                "max": np.max(response_times),
                "p95": np.percentile(response_times, 95),
                "p99": np.percentile(response_times, 99)
            },
            "memory_usage": {
                "mean": np.mean(memory_usage),
                "peak": np.max(memory_usage),
                "std": np.std(memory_usage)
            },
            "cpu_usage": {
                "mean": np.mean(cpu_usage),
                "peak": np.max(cpu_usage),
                "std": np.std(cpu_usage)
            }
        }
        
        return analysis


class BaselineScenario(LoadTestingScenario):
    """Baseline performance testing scenario"""
    
    def __init__(self):
        super().__init__(
            name="baseline",
            description="Baseline performance measurement with single user"
        )
        
    async def run(self, duration_seconds: int = 60, **kwargs) -> Dict:
        """Run baseline scenario"""
        print(f"🔄 Running baseline scenario for {duration_seconds} seconds...")
        
        try:
            load_environment()
            config = Config()
            monitor = ScalabilityMonitor(config)
        except Exception as e:
            return {"error": f"Failed to initialize: {e}"}
        
        # Simulate baseline workload
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        while time.time() < end_time:
            run_start = time.time()
            
            # Simulate processing
            try:
                # Monitor system metrics
                metrics = monitor.get_system_metrics()
                
                # Simulate data processing workload
                data_size_mb = 10  # Small baseline dataset
                self._simulate_data_processing(data_size_mb)
                
                run_time = time.time() - run_start
                
                self.results.append({
                    "timestamp": datetime.now(),
                    "throughput": data_size_mb / run_time if run_time > 0 else 0,
                    "response_time": run_time,
                    "memory_mb": metrics.get("memory_usage_mb", 0),
                    "cpu_percent": metrics.get("cpu_usage_percent", 0),
                    "data_size_mb": data_size_mb,
                    "concurrent_users": 1
                })
                
            except Exception as e:
                self.results.append({
                    "timestamp": datetime.now(),
                    "error": str(e),
                    "concurrent_users": 1
                })
            
            # Small delay between iterations
            await asyncio.sleep(1)
        
        analysis = self.analyze_results()
        print(f"✅ Baseline scenario completed: {analysis['throughput']['mean']:.2f} MB/s average throughput")
        return analysis
    
    def _simulate_data_processing(self, data_size_mb: float):
        """Simulate data processing workload"""
        # Create synthetic data
        rows = int(data_size_mb * 1000)  # Approximate rows
        data = np.random.randn(rows, 5)
        
        # Simulate processing operations
        processed = data * 2
        aggregated = np.mean(processed, axis=0)
        return aggregated


class ConcurrencyScenario(LoadTestingScenario):
    """Concurrency testing scenario with multiple users"""
    
    def __init__(self):
        super().__init__(
            name="concurrency",
            description="Test performance with increasing concurrent users"
        )
        
    async def run(self, duration_seconds: int = 60, max_users: int = 10, **kwargs) -> Dict:
        """Run concurrency scenario"""
        print(f"🔄 Running concurrency scenario with up to {max_users} users for {duration_seconds} seconds...")
        
        try:
            load_environment()
            config = Config()
        except Exception as e:
            return {"error": f"Failed to initialize: {e}"}
        
        # Test with increasing concurrency levels
        for concurrent_users in range(1, max_users + 1, 2):
            print(f"  Testing with {concurrent_users} concurrent users...")
            
            # Run concurrent workload
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
                # Submit tasks
                futures = [
                    executor.submit(self._worker_task, i, duration_seconds // concurrent_users)
                    for i in range(concurrent_users)
                ]
                
                # Wait for completion
                concurrent.futures.wait(futures)
                
                # Collect results
                for future in futures:
                    try:
                        result = future.result()
                        result["concurrent_users"] = concurrent_users
                        self.results.append(result)
                    except Exception as e:
                        self.results.append({
                            "timestamp": datetime.now(),
                            "error": str(e),
                            "concurrent_users": concurrent_users
                        })
        
        analysis = self.analyze_results()
        print(f"✅ Concurrency scenario completed: {analysis['throughput']['mean']:.2f} MB/s average throughput")
        return analysis
    
    def _worker_task(self, worker_id: int, duration: int) -> Dict:
        """Individual worker task"""
        start_time = time.time()
        
        # Get initial metrics
        cpu_start = psutil.cpu_percent()
        memory_start = psutil.virtual_memory().used / (1024 * 1024)
        
        # Simulate work
        data_size_mb = 5 + (worker_id % 3) * 2  # Vary workload size
        self._simulate_concurrent_processing(data_size_mb, duration)
        
        end_time = time.time()
        run_time = end_time - start_time
        
        # Get final metrics
        cpu_end = psutil.cpu_percent()
        memory_end = psutil.virtual_memory().used / (1024 * 1024)
        
        return {
            "timestamp": datetime.now(),
            "worker_id": worker_id,
            "throughput": data_size_mb / run_time if run_time > 0 else 0,
            "response_time": run_time,
            "memory_mb": memory_end - memory_start,
            "cpu_percent": (cpu_start + cpu_end) / 2,
            "data_size_mb": data_size_mb
        }
    
    def _simulate_concurrent_processing(self, data_size_mb: float, duration: int):
        """Simulate concurrent processing workload"""
        end_time = time.time() + duration
        
        while time.time() < end_time:
            # Create and process data
            rows = int(data_size_mb * 500)
            data = np.random.randn(rows, 3)
            
            # Simulate ML operations
            scaled = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
            transformed = np.dot(scaled, np.random.randn(3, 2))
            
            # Small delay
            time.sleep(0.1)


class VolumeScenario(LoadTestingScenario):
    """Data volume testing scenario"""
    
    def __init__(self):
        super().__init__(
            name="volume",
            description="Test performance with increasing data volumes"
        )
        
    async def run(self, duration_seconds: int = 120, **kwargs) -> Dict:
        """Run volume scenario"""
        print(f"🔄 Running volume scenario for {duration_seconds} seconds...")
        
        try:
            load_environment()
            config = Config()
            monitor = ScalabilityMonitor(config)
        except Exception as e:
            return {"error": f"Failed to initialize: {e}"}
        
        # Test with increasing data volumes
        data_sizes = [1, 5, 10, 20, 50, 100, 200]  # MB
        
        for data_size_mb in data_sizes:
            print(f"  Testing with {data_size_mb} MB dataset...")
            
            start_time = time.time()
            
            try:
                # Monitor system before
                metrics_before = monitor.get_system_metrics()
                
                # Process data
                self._process_large_dataset(data_size_mb)
                
                # Monitor system after
                metrics_after = monitor.get_system_metrics()
                
                end_time = time.time()
                run_time = end_time - start_time
                
                self.results.append({
                    "timestamp": datetime.now(),
                    "data_size_mb": data_size_mb,
                    "throughput": data_size_mb / run_time if run_time > 0 else 0,
                    "response_time": run_time,
                    "memory_mb": metrics_after.get("memory_usage_mb", 0) - metrics_before.get("memory_usage_mb", 0),
                    "cpu_percent": (metrics_before.get("cpu_usage_percent", 0) + metrics_after.get("cpu_usage_percent", 0)) / 2,
                    "concurrent_users": 1
                })
                
            except Exception as e:
                self.results.append({
                    "timestamp": datetime.now(),
                    "data_size_mb": data_size_mb,
                    "error": str(e),
                    "concurrent_users": 1
                })
                
                # Stop if we hit memory/processing limits
                if "memory" in str(e).lower() or "out of" in str(e).lower():
                    print(f"  Stopped at {data_size_mb} MB due to resource constraints")
                    break
        
        analysis = self.analyze_results()
        print(f"✅ Volume scenario completed: max {max([r.get('data_size_mb', 0) for r in self.results])} MB processed")
        return analysis
    
    def _process_large_dataset(self, data_size_mb: float):
        """Process large dataset simulation"""
        # Calculate rows based on data size
        rows = int(data_size_mb * 1000 * 10)  # More rows for larger datasets
        cols = 10
        
        print(f"    Processing {rows:,} rows x {cols} columns...")
        
        # Create data in chunks to avoid memory issues
        chunk_size = min(rows, 10000)
        num_chunks = (rows + chunk_size - 1) // chunk_size
        
        results = []
        for i in range(num_chunks):
            start_row = i * chunk_size
            end_row = min((i + 1) * chunk_size, rows)
            chunk_rows = end_row - start_row
            
            # Create chunk
            chunk = np.random.randn(chunk_rows, cols)
            
            # Simulate ML processing
            normalized = (chunk - np.mean(chunk, axis=0)) / (np.std(chunk, axis=0) + 1e-8)
            transformed = np.dot(normalized, np.random.randn(cols, 5))
            aggregated = np.mean(transformed, axis=0)
            
            results.append(aggregated)
        
        # Final aggregation
        final_result = np.mean(results, axis=0)
        return final_result


class StressScenario(LoadTestingScenario):
    """Stress testing scenario to find breaking points"""
    
    def __init__(self):
        super().__init__(
            name="stress",
            description="Stress test to identify system breaking points"
        )
        
    async def run(self, duration_seconds: int = 300, **kwargs) -> Dict:
        """Run stress scenario"""
        print(f"🔄 Running stress scenario for {duration_seconds} seconds...")
        
        # Start with moderate load and gradually increase
        initial_users = 2
        max_users = multiprocessing.cpu_count() * 2
        user_increment = 2
        increment_interval = 30  # seconds
        
        current_users = initial_users
        start_time = time.time()
        last_increment = start_time
        
        # Track active workers
        active_workers = []
        
        while time.time() - start_time < duration_seconds:
            current_time = time.time()
            
            # Increase load periodically
            if current_time - last_increment >= increment_interval and current_users < max_users:
                current_users = min(current_users + user_increment, max_users)
                last_increment = current_time
                print(f"  Increasing load to {current_users} concurrent users...")
            
            # Start new workers if needed
            while len(active_workers) < current_users:
                worker = threading.Thread(
                    target=self._stress_worker,
                    args=(len(active_workers), current_time)
                )
                worker.daemon = True
                worker.start()
                active_workers.append(worker)
            
            # Check system health
            memory_percent = psutil.virtual_memory().percent
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if memory_percent > 90 or cpu_percent > 95:
                print(f"  ⚠️ High resource usage detected (Memory: {memory_percent:.1f}%, CPU: {cpu_percent:.1f}%)")
                
                # Record stress point
                self.results.append({
                    "timestamp": datetime.now(),
                    "stress_point": True,
                    "concurrent_users": current_users,
                    "memory_percent": memory_percent,
                    "cpu_percent": cpu_percent,
                    "duration_seconds": current_time - start_time
                })
                
                if memory_percent > 95:
                    print(f"  🚨 Critical memory usage reached, stopping stress test")
                    break
            
            await asyncio.sleep(5)
        
        # Clean up workers
        for worker in active_workers:
            if worker.is_alive():
                try:
                    worker.join(timeout=5)
                except:
                    pass
        
        analysis = self.analyze_results()
        max_users_reached = max([r.get('concurrent_users', 0) for r in self.results])
        print(f"✅ Stress scenario completed: max {max_users_reached} concurrent users reached")
        return analysis
    
    def _stress_worker(self, worker_id: int, start_time: float):
        """Individual stress test worker"""
        try:
            while True:
                iteration_start = time.time()
                
                # Get system metrics
                memory_usage = psutil.virtual_memory().percent
                cpu_usage = psutil.cpu_percent()
                
                # Simulate intensive work
                data_size = 5 + (worker_id % 5)  # 5-10 MB per worker
                self._intensive_computation(data_size)
                
                iteration_time = time.time() - iteration_start
                
                # Record result
                self.results.append({
                    "timestamp": datetime.now(),
                    "worker_id": worker_id,
                    "throughput": data_size / iteration_time if iteration_time > 0 else 0,
                    "response_time": iteration_time,
                    "memory_percent": memory_usage,
                    "cpu_percent": cpu_usage,
                    "data_size_mb": data_size,
                    "duration_from_start": time.time() - start_time
                })
                
                # Check if we should stop (high resource usage)
                if memory_usage > 90:
                    break
                    
                # Small delay to prevent overwhelming
                time.sleep(0.5)
                
        except Exception as e:
            self.results.append({
                "timestamp": datetime.now(),
                "worker_id": worker_id,
                "error": str(e)
            })
    
    def _intensive_computation(self, data_size_mb: float):
        """Intensive computation to stress the system"""
        rows = int(data_size_mb * 1000)
        cols = 20
        
        # Create large arrays
        data1 = np.random.randn(rows, cols)
        data2 = np.random.randn(rows, cols)
        
        # Intensive operations
        result = np.dot(data1.T, data2)
        eigenvals = np.linalg.eigvals(result)
        processed = np.sort(eigenvals)
        
        return processed


class RegressionDetectionScenario(LoadTestingScenario):
    """Performance regression detection scenario"""
    
    def __init__(self, baseline_results: Optional[Dict] = None):
        super().__init__(
            name="regression",
            description="Detect performance regressions compared to baseline"
        )
        self.baseline_results = baseline_results
        
    async def run(self, duration_seconds: int = 60, **kwargs) -> Dict:
        """Run regression detection scenario"""
        print(f"🔄 Running regression detection scenario for {duration_seconds} seconds...")
        
        if not self.baseline_results:
            print("  No baseline results provided, collecting new baseline...")
            baseline_scenario = BaselineScenario()
            self.baseline_results = await baseline_scenario.run(duration_seconds // 2)
        
        # Run current performance test
        current_scenario = BaselineScenario()
        current_results = await current_scenario.run(duration_seconds // 2)
        
        # Compare results
        regression_analysis = self._detect_regressions(self.baseline_results, current_results)
        
        # Store comparison results
        self.results = [
            {
                "timestamp": datetime.now(),
                "regression_analysis": regression_analysis,
                "baseline_throughput": self.baseline_results.get("throughput", {}).get("mean", 0),
                "current_throughput": current_results.get("throughput", {}).get("mean", 0),
                "throughput_change_percent": regression_analysis.get("throughput_change_percent", 0),
                "regression_detected": regression_analysis.get("regression_detected", False)
            }
        ]
        
        if regression_analysis.get("regression_detected", False):
            print(f"  🚨 Performance regression detected!")
            for metric, change in regression_analysis.get("significant_changes", {}).items():
                print(f"    {metric}: {change:.1f}% change")
        else:
            print(f"  ✅ No significant performance regression detected")
        
        return regression_analysis
    
    def _detect_regressions(self, baseline: Dict, current: Dict) -> Dict:
        """Detect performance regressions"""
        regression_threshold = 10  # 10% degradation threshold
        
        analysis = {
            "regression_detected": False,
            "significant_changes": {},
            "comparison_timestamp": datetime.now()
        }
        
        # Compare key metrics
        metrics_to_compare = ["throughput", "response_time", "memory_usage", "cpu_usage"]
        
        for metric in metrics_to_compare:
            baseline_value = baseline.get(metric, {}).get("mean", 0)
            current_value = current.get(metric, {}).get("mean", 0)
            
            if baseline_value > 0:
                change_percent = ((current_value - baseline_value) / baseline_value) * 100
                analysis[f"{metric}_change_percent"] = change_percent
                
                # Check for regression (degradation)
                if metric in ["response_time", "memory_usage", "cpu_usage"]:
                    # Higher is worse for these metrics
                    if change_percent > regression_threshold:
                        analysis["regression_detected"] = True
                        analysis["significant_changes"][metric] = change_percent
                else:
                    # Lower is worse for throughput
                    if change_percent < -regression_threshold:
                        analysis["regression_detected"] = True
                        analysis["significant_changes"][metric] = change_percent
        
        return analysis


class LoadTestRunner:
    """Main load testing runner"""
    
    def __init__(self):
        try:
            load_environment()
            self.config = Config()
            self.logger = Logger().get_logger()
        except Exception as e:
            print(f"Failed to initialize: {e}")
            self.config = None
            self.logger = None
        
        self.scenarios = {
            "baseline": BaselineScenario(),
            "concurrency": ConcurrencyScenario(),
            "volume": VolumeScenario(),
            "stress": StressScenario(),
            "regression": RegressionDetectionScenario()
        }
        
        self.results_dir = pipeline_root / "results" / "load_testing"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    async def run_scenario(self, scenario_name: str, **kwargs) -> Dict:
        """Run a specific load testing scenario"""
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list(self.scenarios.keys())}")
        
        scenario = self.scenarios[scenario_name]
        print(f"\n{'='*50}")
        print(f"🚀 Starting scenario: {scenario.name}")
        print(f"📝 Description: {scenario.description}")
        print(f"{'='*50}")
        
        start_time = time.time()
        
        try:
            # Run scenario
            results = await scenario.run(**kwargs)
            
            # Save results
            self._save_results(scenario_name, results, scenario.results)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            print(f"\n✅ Scenario '{scenario.name}' completed in {total_time:.1f} seconds")
            return results
            
        except Exception as e:
            print(f"\n❌ Scenario '{scenario.name}' failed: {e}")
            raise
    
    async def run_all_scenarios(self, duration_per_scenario: int = 60) -> Dict:
        """Run all load testing scenarios"""
        all_results = {}
        
        # Run scenarios in order
        scenario_order = ["baseline", "volume", "concurrency", "stress"]
        
        for scenario_name in scenario_order:
            try:
                results = await self.run_scenario(scenario_name, duration_seconds=duration_per_scenario)
                all_results[scenario_name] = results
                
                # Brief pause between scenarios
                await asyncio.sleep(10)
                
            except Exception as e:
                print(f"Failed to run scenario {scenario_name}: {e}")
                all_results[scenario_name] = {"error": str(e)}
        
        # Run regression detection with baseline
        if "baseline" in all_results and not all_results["baseline"].get("error"):
            try:
                regression_scenario = RegressionDetectionScenario(all_results["baseline"])
                regression_results = await regression_scenario.run(duration_seconds=60)
                all_results["regression"] = regression_results
            except Exception as e:
                all_results["regression"] = {"error": str(e)}
        
        # Generate summary report
        summary = self._generate_summary_report(all_results)
        all_results["summary"] = summary
        
        # Save comprehensive results
        self._save_comprehensive_results(all_results)
        
        return all_results
    
    def _save_results(self, scenario_name: str, analysis: Dict, raw_results: List[Dict]):
        """Save scenario results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save analysis
        analysis_file = self.results_dir / f"{scenario_name}_analysis_{timestamp}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Save raw results
        if raw_results:
            raw_file = self.results_dir / f"{scenario_name}_raw_{timestamp}.json"
            with open(raw_file, 'w') as f:
                json.dump(raw_results, f, indent=2, default=str)
            
            # Save as CSV if possible
            try:
                df = pd.DataFrame(raw_results)
                csv_file = self.results_dir / f"{scenario_name}_raw_{timestamp}.csv"
                df.to_csv(csv_file, index=False)
            except Exception as e:
                print(f"Could not save CSV: {e}")
        
        print(f"📊 Results saved to: {analysis_file}")
    
    def _save_comprehensive_results(self, all_results: Dict):
        """Save comprehensive results from all scenarios"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save comprehensive JSON
        comprehensive_file = self.results_dir / f"comprehensive_load_test_{timestamp}.json"
        with open(comprehensive_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Generate markdown report
        report_file = self.results_dir / f"load_test_report_{timestamp}.md"
        self._generate_markdown_report(all_results, report_file)
        
        print(f"📈 Comprehensive results saved to: {comprehensive_file}")
        print(f"📋 Report saved to: {report_file}")
    
    def _generate_summary_report(self, all_results: Dict) -> Dict:
        """Generate summary report from all scenarios"""
        summary = {
            "test_timestamp": datetime.now(),
            "scenarios_run": len([k for k, v in all_results.items() if not v.get("error")]),
            "scenarios_failed": len([k for k, v in all_results.items() if v.get("error")]),
            "key_metrics": {},
            "recommendations": []
        }
        
        # Extract key metrics from successful scenarios
        if "baseline" in all_results and not all_results["baseline"].get("error"):
            baseline = all_results["baseline"]
            summary["key_metrics"]["baseline_throughput"] = baseline.get("throughput", {}).get("mean", 0)
            summary["key_metrics"]["baseline_response_time"] = baseline.get("response_time", {}).get("mean", 0)
        
        if "stress" in all_results and not all_results["stress"].get("error"):
            stress_results = all_results["stress"]["results"] if "results" in all_results["stress"] else []
            if stress_results:
                max_users = max([r.get("concurrent_users", 0) for r in stress_results])
                summary["key_metrics"]["max_concurrent_users"] = max_users
        
        # Generate recommendations
        if summary["key_metrics"].get("baseline_throughput", 0) < 1:
            summary["recommendations"].append("Consider optimizing data processing pipeline for better throughput")
        
        if summary["key_metrics"].get("max_concurrent_users", 0) < 10:
            summary["recommendations"].append("System may benefit from horizontal scaling improvements")
        
        return summary
    
    def _generate_markdown_report(self, all_results: Dict, report_file: Path):
        """Generate markdown report"""
        with open(report_file, 'w') as f:
            f.write("# Load Testing Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            if "summary" in all_results:
                summary = all_results["summary"]
                f.write("## Summary\n\n")
                f.write(f"- **Scenarios Run:** {summary.get('scenarios_run', 0)}\n")
                f.write(f"- **Scenarios Failed:** {summary.get('scenarios_failed', 0)}\n")
                
                if "key_metrics" in summary:
                    f.write("\n### Key Metrics\n\n")
                    for metric, value in summary["key_metrics"].items():
                        f.write(f"- **{metric.replace('_', ' ').title()}:** {value:.2f}\n")
                
                if "recommendations" in summary:
                    f.write("\n### Recommendations\n\n")
                    for rec in summary["recommendations"]:
                        f.write(f"- {rec}\n")
            
            # Individual scenarios
            f.write("\n## Scenario Results\n\n")
            
            for scenario_name, results in all_results.items():
                if scenario_name == "summary":
                    continue
                
                f.write(f"### {scenario_name.title()} Scenario\n\n")
                
                if results.get("error"):
                    f.write(f"**Status:** ❌ Failed\n")
                    f.write(f"**Error:** {results['error']}\n\n")
                else:
                    f.write(f"**Status:** ✅ Completed\n\n")
                    
                    if "throughput" in results:
                        throughput = results["throughput"]
                        f.write(f"**Throughput:**\n")
                        f.write(f"- Mean: {throughput.get('mean', 0):.2f} MB/s\n")
                        f.write(f"- Peak: {throughput.get('max', 0):.2f} MB/s\n")
                        f.write(f"- P95: {throughput.get('p95', 0):.2f} MB/s\n\n")
                    
                    if "response_time" in results:
                        response_time = results["response_time"]
                        f.write(f"**Response Time:**\n")
                        f.write(f"- Mean: {response_time.get('mean', 0):.2f}s\n")
                        f.write(f"- P95: {response_time.get('p95', 0):.2f}s\n")
                        f.write(f"- P99: {response_time.get('p99', 0):.2f}s\n\n")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="ML Pipeline Load Testing")
    parser.add_argument(
        "--scenario",
        choices=["baseline", "concurrency", "volume", "stress", "regression", "all"],
        default="baseline",
        help="Load testing scenario to run"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Duration in seconds for each scenario"
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=10,
        help="Maximum concurrent users (for concurrency scenario)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = LoadTestRunner()
    
    if args.output_dir:
        runner.results_dir = Path(args.output_dir)
        runner.results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🔬 ML Pipeline Load Testing")
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Results Directory: {runner.results_dir}")
    
    try:
        if args.scenario == "all":
            results = await runner.run_all_scenarios(duration_per_scenario=args.duration)
        else:
            kwargs = {"duration_seconds": args.duration}
            if args.scenario == "concurrency":
                kwargs["max_users"] = args.max_users
            
            results = await runner.run_scenario(args.scenario, **kwargs)
        
        print(f"\n🎉 Load testing completed successfully!")
        return results
        
    except Exception as e:
        print(f"\n💥 Load testing failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())