#!/usr/bin/env python3
"""
Evaluation Framework for Multi-Mode Scalability Analysis
Provides unified evaluation metrics, experiment matrix, and comparative analysis

Part of Data Engineering at Scale project - Phase 4 implementation
"""

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

from src.utils import Logger


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for mode comparison"""
    
    # Training performance
    training_time_seconds: float = 0.0
    training_time_per_1k_rows_ms: float = 0.0
    throughput_records_per_second: float = 0.0
    
    # Memory efficiency
    peak_memory_usage_mb: float = 0.0
    memory_per_1k_rows_mb: float = 0.0
    memory_efficiency_score: float = 0.0
    
    # Cost efficiency (memory × time)
    cost_efficiency_score: float = 0.0  # Lower is better
    cost_per_1k_rows: float = 0.0
    
    # Model accuracy
    mae: float = 0.0
    mse: float = 0.0
    rmse: float = 0.0
    r2_score: float = 0.0
    
    # Resource utilization
    cpu_utilization_percent: float = 0.0
    resource_efficiency_score: float = 0.0
    
    # Scalability indicators
    linear_scalability_score: float = 0.0
    performance_degradation_factor: float = 1.0
    
    # Data characteristics
    total_rows: int = 0
    data_size_mb: float = 0.0
    feature_count: int = 0


@dataclass
class ExperimentResult:
    """Single experiment result with complete metadata"""
    
    mode: str = ""
    volume: str = ""
    timestamp: str = ""
    metrics: EvaluationMetrics = None
    mode_specific_data: Dict = None
    status: str = "UNKNOWN"  # SUCCESS, FAILED, PARTIAL
    error_message: str = ""
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = EvaluationMetrics()
        if self.mode_specific_data is None:
            self.mode_specific_data = {}


class EvaluationFramework:
    """
    Comprehensive evaluation framework for multi-mode scalability analysis
    """
    
    def __init__(self, base_results_dir: str = "results/evaluation"):
        self.base_results_dir = Path(base_results_dir)
        self.base_results_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = Logger().get_logger()
        
        # Evaluation configuration
        self.volumes = ["small", "medium", "large", "full"]
        self.modes = ["spark_gbt", "spark_rf", "hybrid_sklearn"]
        
        # Results storage
        self.experiment_results: Dict[str, ExperimentResult] = {}
        self.comparative_analysis = {}
        
        self.logger.info("📊 Evaluation Framework initialized")
    
    def load_experiment_result(self, result_file: str) -> Optional[ExperimentResult]:
        """Load and parse experiment result from volume test files"""
        
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Extract experiment metadata
            volume = data.get("volume", "unknown")
            modes_data = data.get("mode_results", {})
            
            results = []
            
            for mode, mode_data in modes_data.items():
                if "error" in mode_data:
                    result = ExperimentResult(
                        mode=mode,
                        volume=volume,
                        timestamp=mode_data.get("test_timestamp", ""),
                        status="FAILED",
                        error_message=mode_data["error"]
                    )
                else:
                    # Extract metrics from the result
                    metrics = self._extract_metrics(mode_data)
                    
                    result = ExperimentResult(
                        mode=mode,
                        volume=volume,
                        timestamp=mode_data.get("test_timestamp", ""),
                        metrics=metrics,
                        mode_specific_data=mode_data.get("mode_specific_monitoring", {}),
                        status="SUCCESS"
                    )
                
                results.append(result)
                
                # Store in experiment results
                key = f"{mode}_{volume}"
                self.experiment_results[key] = result
            
            self.logger.info(f"📊 Loaded {len(results)} experiment results from {result_file}")
            return results
        
        except Exception as e:
            self.logger.error(f"❌ Failed to load experiment result: {e}")
            return None
    
    def _extract_metrics(self, mode_data: Dict) -> EvaluationMetrics:
        """Extract standardized metrics from mode-specific data"""
        
        metrics = EvaluationMetrics()
        
        try:
            # Training performance
            training_result = mode_data.get("training_result", {})
            performance_summary = mode_data.get("performance_summary", {})
            
            metrics.training_time_seconds = performance_summary.get("training_time", 0.0)
            metrics.total_rows = performance_summary.get("data_rows_processed", 0)
            
            # Calculate training time per 1k rows
            if metrics.total_rows > 0:
                metrics.training_time_per_1k_rows_ms = (metrics.training_time_seconds * 1000) / (metrics.total_rows / 1000)
                metrics.throughput_records_per_second = metrics.total_rows / max(metrics.training_time_seconds, 0.1)
            
            # Memory efficiency (handle both object and string formats)
            scalability_metrics = training_result.get("scalability_metrics")
            if scalability_metrics:
                if isinstance(scalability_metrics, str):
                    # Parse string representation
                    scalability_dict = self._parse_scalability_metrics_string(scalability_metrics)
                    metrics.peak_memory_usage_mb = scalability_dict.get('peak_memory_usage_mb', 0.0)
                    metrics.memory_efficiency_score = scalability_dict.get('resource_efficiency_score', 0.0)
                    metrics.linear_scalability_score = scalability_dict.get('linear_scalability_score', 0.0)
                    
                    # Extract throughput from parsed metrics
                    metrics.throughput_records_per_second = scalability_dict.get('throughput_records_per_second', 0.0)
                    if metrics.throughput_records_per_second == 0.0 and metrics.training_time_seconds > 0:
                        metrics.throughput_records_per_second = metrics.total_rows / metrics.training_time_seconds
                    
                else:
                    # Handle object format (fallback to original logic)
                    metrics.peak_memory_usage_mb = getattr(scalability_metrics, 'peak_memory_usage_mb', 0.0)
                    metrics.memory_efficiency_score = getattr(scalability_metrics, 'resource_efficiency_score', 0.0)
                    metrics.linear_scalability_score = getattr(scalability_metrics, 'linear_scalability_score', 0.0)
                    metrics.throughput_records_per_second = getattr(scalability_metrics, 'throughput_records_per_second', 0.0)
                
                if metrics.total_rows > 0:
                    metrics.memory_per_1k_rows_mb = metrics.peak_memory_usage_mb / (metrics.total_rows / 1000)
            
            # Cost efficiency (memory × time) - improved calculation  
            if metrics.peak_memory_usage_mb > 0 and metrics.training_time_seconds > 0:
                metrics.cost_efficiency_score = metrics.peak_memory_usage_mb * metrics.training_time_seconds
            else:
                metrics.cost_efficiency_score = 0.0
                
            if metrics.total_rows > 0:
                metrics.cost_per_1k_rows = metrics.cost_efficiency_score / (metrics.total_rows / 1000)
            else:
                metrics.cost_per_1k_rows = 0.0
            
            # Model accuracy
            test_metrics = training_result.get("metrics", {}).get("test_metrics", {})
            metrics.mae = test_metrics.get("mae", 0.0)
            metrics.mse = test_metrics.get("mse", 0.0) 
            metrics.rmse = test_metrics.get("rmse", 0.0)
            metrics.r2_score = test_metrics.get("r2", 0.0)
            
            # Resource utilization
            monitoring_result = mode_data.get("mode_specific_monitoring", {})
            if "resource_samples" in monitoring_result:
                samples = monitoring_result["resource_samples"]
                if samples:
                    cpu_values = [s.get("cpu_percent", 0) for s in samples]
                    metrics.cpu_utilization_percent = np.mean(cpu_values)
            
            # Calculate resource efficiency score
            metrics.resource_efficiency_score = self._calculate_resource_efficiency(metrics)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Metrics extraction warning: {e}")
        
        return metrics
    
    def _calculate_resource_efficiency(self, metrics: EvaluationMetrics) -> float:
        """Calculate overall resource efficiency score (0.0 to 1.0, higher is better)"""
        
        try:
            # Normalize components to 0-1 scale (higher is better)
            
            # Throughput efficiency (normalize to reasonable max)
            max_throughput = 1000  # records/second
            throughput_score = min(metrics.throughput_records_per_second / max_throughput, 1.0)
            
            # Memory efficiency (inverse of memory usage, normalize to reasonable range)
            max_memory_per_1k = 100  # MB per 1k rows
            memory_score = max(0.0, 1.0 - (metrics.memory_per_1k_rows_mb / max_memory_per_1k))
            
            # Model accuracy (R² can be negative, so clamp to 0-1)
            accuracy_score = max(0.0, min(metrics.r2_score, 1.0))
            
            # Weighted combination
            efficiency_score = (0.4 * throughput_score + 
                              0.3 * memory_score + 
                              0.3 * accuracy_score)
            
            return efficiency_score
        
        except:
            return 0.0
    
    def run_experiment_matrix_evaluation(self, volume_test_results_dir: str = "results/volume_tests") -> Dict:
        """Run comprehensive experiment matrix evaluation"""
        
        self.logger.info("📈 Starting experiment matrix evaluation")
        
        # Load all experiment results
        results_dir = Path(volume_test_results_dir)
        if not results_dir.exists():
            self.logger.error(f"❌ Results directory not found: {results_dir}")
            return {}
        
        # Find all result files
        result_files = list(results_dir.glob("*_volume_test_*.json"))
        
        if not result_files:
            self.logger.warning("⚠️ No volume test results found")
            return {}
        
        self.logger.info(f"📊 Found {len(result_files)} result files")
        
        # Load all results
        all_results = []
        for result_file in result_files:
            results = self.load_experiment_result(result_file)
            if results:
                all_results.extend(results)
        
        # Perform comprehensive analysis
        analysis = {
            "experiment_matrix": self._create_experiment_matrix(),
            "scalability_analysis": self._analyze_scalability_trends(),
            "comparative_analysis": self._create_comparative_analysis(),
            "cost_analysis": self._analyze_cost_efficiency(),
            "performance_recommendations": self._generate_recommendations(),
            "evaluation_metadata": {
                "total_experiments": len(self.experiment_results),
                "evaluation_timestamp": datetime.now().isoformat(),
                "volumes_tested": list(set(r.volume for r in all_results)),
                "modes_tested": list(set(r.mode for r in all_results))
            }
        }
        
        # Save comprehensive evaluation
        self._save_evaluation_results(analysis)
        
        self.logger.info("✅ Experiment matrix evaluation completed")
        return analysis
    
    def _create_experiment_matrix(self) -> Dict:
        """Create experiment matrix with all mode×volume combinations"""
        
        matrix = {
            "dimensions": {
                "modes": self.modes,
                "volumes": self.volumes
            },
            "results": {},
            "coverage": {
                "total_combinations": len(self.modes) * len(self.volumes),
                "completed_combinations": 0,
                "coverage_percentage": 0.0
            }
        }
        
        # Fill matrix with results
        for mode in self.modes:
            matrix["results"][mode] = {}
            for volume in self.volumes:
                key = f"{mode}_{volume}"
                
                if key in self.experiment_results:
                    result = self.experiment_results[key]
                    matrix["results"][mode][volume] = {
                        "status": result.status,
                        "metrics": asdict(result.metrics) if result.metrics else None,
                        "timestamp": result.timestamp
                    }
                    if result.status == "SUCCESS":
                        matrix["coverage"]["completed_combinations"] += 1
                else:
                    matrix["results"][mode][volume] = {
                        "status": "NOT_RUN",
                        "metrics": None,
                        "timestamp": None
                    }
        
        # Calculate coverage
        matrix["coverage"]["coverage_percentage"] = (
            matrix["coverage"]["completed_combinations"] / 
            matrix["coverage"]["total_combinations"] * 100
        )
        
        return matrix
    
    def _analyze_scalability_trends(self) -> Dict:
        """Analyze scalability trends across data volumes"""
        
        analysis = {
            "mode_scalability": {},
            "volume_scaling_factors": {},
            "linear_scalability_scores": {},
            "bottleneck_analysis": {}
        }
        
        for mode in self.modes:
            mode_results = []
            
            for volume in self.volumes:
                key = f"{mode}_{volume}"
                if key in self.experiment_results and self.experiment_results[key].status == "SUCCESS":
                    mode_results.append(self.experiment_results[key])
            
            if len(mode_results) >= 2:
                # Analyze scaling trends for this mode
                analysis["mode_scalability"][mode] = self._analyze_mode_scaling(mode_results)
        
        return analysis
    
    def _analyze_mode_scaling(self, mode_results: List[ExperimentResult]) -> Dict:
        """Analyze scaling behavior for a specific mode"""
        
        # Sort by data volume (assuming small < medium < large < full)
        volume_order = {"small": 1, "medium": 2, "large": 3, "full": 4}
        mode_results.sort(key=lambda r: volume_order.get(r.volume, 999))
        
        scaling_analysis = {
            "training_time_scaling": [],
            "memory_scaling": [],
            "throughput_scaling": [],
            "linear_scalability_score": 0.0,
            "scaling_trend": "UNKNOWN"
        }
        
        # Extract metrics for analysis
        volumes = [r.volume for r in mode_results]
        training_times = [r.metrics.training_time_seconds for r in mode_results]
        memory_usage = [r.metrics.peak_memory_usage_mb for r in mode_results]
        throughput = [r.metrics.throughput_records_per_second for r in mode_results]
        row_counts = [r.metrics.total_rows for r in mode_results]
        
        # Calculate scaling factors
        base_time = training_times[0]
        base_memory = memory_usage[0]
        base_rows = row_counts[0]
        
        for i, result in enumerate(mode_results):
            if base_rows > 0 and base_time > 0:
                data_scale_factor = row_counts[i] / base_rows
                time_scale_factor = training_times[i] / base_time
                memory_scale_factor = memory_usage[i] / base_memory if base_memory > 0 else 1.0
                
                scaling_analysis["training_time_scaling"].append({
                    "volume": volumes[i],
                    "data_scale_factor": data_scale_factor,
                    "time_scale_factor": time_scale_factor,
                    "scaling_efficiency": data_scale_factor / time_scale_factor if time_scale_factor > 0 else 0.0
                })
                
                scaling_analysis["memory_scaling"].append({
                    "volume": volumes[i],
                    "data_scale_factor": data_scale_factor,
                    "memory_scale_factor": memory_scale_factor,
                    "memory_efficiency": data_scale_factor / memory_scale_factor if memory_scale_factor > 0 else 0.0
                })
        
        # Calculate linear scalability score
        if len(training_times) >= 2:
            # Linear scalability means time scales linearly with data size
            scaling_analysis["linear_scalability_score"] = self._calculate_linear_scalability(
                row_counts, training_times
            )
        
        # Determine scaling trend
        if len(scaling_analysis["training_time_scaling"]) >= 2:
            efficiency_trend = [s["scaling_efficiency"] for s in scaling_analysis["training_time_scaling"]]
            if all(e >= 0.8 for e in efficiency_trend):
                scaling_analysis["scaling_trend"] = "LINEAR"
            elif efficiency_trend[-1] < efficiency_trend[0] * 0.5:
                scaling_analysis["scaling_trend"] = "DEGRADING"
            else:
                scaling_analysis["scaling_trend"] = "SUB_LINEAR"
        
        return scaling_analysis
    
    def _calculate_linear_scalability(self, row_counts: List[int], training_times: List[float]) -> float:
        """Calculate linear scalability score (1.0 = perfect linear scaling)"""
        
        if len(row_counts) < 2 or len(training_times) < 2:
            return 0.0
        
        try:
            # Calculate expected vs actual scaling
            base_rows, base_time = row_counts[0], training_times[0]
            
            deviations = []
            for i in range(1, len(row_counts)):
                expected_scale = row_counts[i] / base_rows
                actual_scale = training_times[i] / base_time
                
                # Deviation from linear scaling
                deviation = abs(expected_scale - actual_scale) / expected_scale
                deviations.append(deviation)
            
            # Score is inverse of average deviation
            avg_deviation = np.mean(deviations)
            linear_score = max(0.0, 1.0 - avg_deviation)
            
            return linear_score
        
        except:
            return 0.0
    
    def _create_comparative_analysis(self) -> Dict:
        """Create comprehensive comparative analysis across modes"""
        
        comparison = {
            "performance_leaders": {},
            "mode_rankings": {},
            "volume_specific_winners": {},
            "overall_recommendations": {}
        }
        
        # Find performance leaders in different categories
        categories = [
            ("fastest_training", lambda m: -m.training_time_seconds),
            ("most_memory_efficient", lambda m: -m.memory_per_1k_rows_mb),
            ("best_accuracy", lambda m: m.r2_score),
            ("best_cost_efficiency", lambda m: -m.cost_efficiency_score),
            ("highest_throughput", lambda m: m.throughput_records_per_second)
        ]
        
        for category, score_func in categories:
            leaders = {}
            
            for volume in self.volumes:
                volume_scores = []
                
                for mode in self.modes:
                    key = f"{mode}_{volume}"
                    if key in self.experiment_results and self.experiment_results[key].status == "SUCCESS":
                        metrics = self.experiment_results[key].metrics
                        score = score_func(metrics)
                        volume_scores.append((mode, score, metrics))
                
                if volume_scores:
                    # Sort by score (descending)
                    volume_scores.sort(key=lambda x: x[1], reverse=True)
                    leaders[volume] = {
                        "mode": volume_scores[0][0],
                        "score": volume_scores[0][1],
                        "all_scores": [(mode, score) for mode, score, _ in volume_scores]
                    }
            
            comparison["performance_leaders"][category] = leaders
        
        # Create overall mode rankings
        for volume in self.volumes:
            mode_scores = {}
            
            for mode in self.modes:
                key = f"{mode}_{volume}"
                if key in self.experiment_results and self.experiment_results[key].status == "SUCCESS":
                    metrics = self.experiment_results[key].metrics
                    
                    # Weighted overall score
                    score = (
                        0.3 * metrics.resource_efficiency_score +  # Resource efficiency
                        0.2 * max(0.0, min(metrics.r2_score, 1.0)) +  # Accuracy (clamped)
                        0.2 * (1.0 / (1.0 + metrics.cost_per_1k_rows / 1000)) +  # Cost efficiency
                        0.2 * metrics.linear_scalability_score +  # Scalability
                        0.1 * min(metrics.throughput_records_per_second / 100, 1.0)  # Throughput
                    )
                    
                    mode_scores[mode] = score
            
            if mode_scores:
                # Rank modes for this volume
                ranked_modes = sorted(mode_scores.items(), key=lambda x: x[1], reverse=True)
                comparison["mode_rankings"][volume] = ranked_modes
        
        return comparison
    
    def _analyze_cost_efficiency(self) -> Dict:
        """Analyze cost efficiency across modes and volumes"""
        
        cost_analysis = {
            "cost_per_volume": {},
            "cost_scaling_analysis": {},
            "cost_recommendations": {}
        }
        
        # Analyze cost for each volume
        for volume in self.volumes:
            volume_costs = {}
            
            for mode in self.modes:
                key = f"{mode}_{volume}"
                if key in self.experiment_results and self.experiment_results[key].status == "SUCCESS":
                    metrics = self.experiment_results[key].metrics
                    
                    volume_costs[mode] = {
                        "total_cost": metrics.cost_efficiency_score,
                        "cost_per_1k_rows": metrics.cost_per_1k_rows,
                        "memory_component": metrics.peak_memory_usage_mb,
                        "time_component": metrics.training_time_seconds
                    }
            
            if volume_costs:
                # Find most cost-effective mode for this volume
                best_mode = min(volume_costs.items(), key=lambda x: x[1]["cost_per_1k_rows"])
                cost_analysis["cost_per_volume"][volume] = {
                    "all_costs": volume_costs,
                    "most_cost_effective": {
                        "mode": best_mode[0],
                        "cost_per_1k_rows": best_mode[1]["cost_per_1k_rows"]
                    }
                }
        
        return cost_analysis
    
    def _generate_recommendations(self) -> Dict:
        """Generate performance and deployment recommendations"""
        
        recommendations = {
            "deployment_scenarios": {},
            "optimization_suggestions": {},
            "production_readiness": {}
        }
        
        # Deployment scenario recommendations
        scenarios = {
            "quick_prototyping": {
                "priority": "speed",
                "description": "Fast development and testing"
            },
            "production_high_throughput": {
                "priority": "throughput", 
                "description": "High-volume production workloads"
            },
            "cost_constrained": {
                "priority": "cost",
                "description": "Budget-conscious deployments"
            },
            "accuracy_critical": {
                "priority": "accuracy",
                "description": "High-accuracy requirements"
            }
        }
        
        for scenario, config in scenarios.items():
            best_mode = self._find_best_mode_for_scenario(config["priority"])
            recommendations["deployment_scenarios"][scenario] = {
                "recommended_mode": best_mode["mode"] if best_mode else "unknown",
                "reason": best_mode["reason"] if best_mode else "insufficient data",
                "description": config["description"]
            }
        
        # Production readiness assessment
        for mode in self.modes:
            readiness = self._assess_production_readiness(mode)
            recommendations["production_readiness"][mode] = readiness
        
        return recommendations
    
    def _find_best_mode_for_scenario(self, priority: str) -> Optional[Dict]:
        """Find best mode for specific deployment scenario"""
        
        # Get results for largest volume tested (most representative)
        volume_order = ["full", "large", "medium", "small"]
        target_volume = None
        
        for vol in volume_order:
            if any(f"{mode}_{vol}" in self.experiment_results for mode in self.modes):
                target_volume = vol
                break
        
        if not target_volume:
            return None
        
        mode_scores = {}
        
        for mode in self.modes:
            key = f"{mode}_{target_volume}"
            if key in self.experiment_results and self.experiment_results[key].status == "SUCCESS":
                metrics = self.experiment_results[key].metrics
                
                if priority == "speed":
                    score = 1.0 / (1.0 + metrics.training_time_seconds / 3600)  # Normalize to hours
                elif priority == "throughput":
                    score = metrics.throughput_records_per_second
                elif priority == "cost":
                    score = 1.0 / (1.0 + metrics.cost_per_1k_rows / 1000)
                elif priority == "accuracy":
                    score = max(0.0, metrics.r2_score)
                else:
                    score = metrics.resource_efficiency_score
                
                mode_scores[mode] = score
        
        if not mode_scores:
            return None
        
        best_mode = max(mode_scores.items(), key=lambda x: x[1])
        
        return {
            "mode": best_mode[0],
            "score": best_mode[1],
            "reason": f"Best {priority} performance for {target_volume} volume"
        }
    
    def _assess_production_readiness(self, mode: str) -> Dict:
        """Assess production readiness for a mode"""
        
        assessment = {
            "overall_score": 0.0,
            "readiness_level": "NOT_READY",
            "strengths": [],
            "concerns": [],
            "recommendations": []
        }
        
        # Collect all results for this mode
        mode_results = [r for k, r in self.experiment_results.items() 
                       if k.startswith(f"{mode}_") and r.status == "SUCCESS"]
        
        if not mode_results:
            assessment["concerns"].append("No successful test results available")
            return assessment
        
        # Analyze across different criteria
        criteria_scores = []
        
        # Performance consistency
        training_times = [r.metrics.training_time_seconds for r in mode_results]
        if len(training_times) > 1:
            time_variance = np.std(training_times) / np.mean(training_times)
            consistency_score = max(0.0, 1.0 - time_variance)
            criteria_scores.append(consistency_score)
            
            if consistency_score > 0.8:
                assessment["strengths"].append("Consistent performance across volumes")
            elif consistency_score < 0.5:
                assessment["concerns"].append("High performance variance")
        
        # Resource efficiency
        efficiency_scores = [r.metrics.resource_efficiency_score for r in mode_results]
        avg_efficiency = np.mean(efficiency_scores)
        criteria_scores.append(avg_efficiency)
        
        if avg_efficiency > 0.7:
            assessment["strengths"].append("High resource efficiency")
        elif avg_efficiency < 0.4:
            assessment["concerns"].append("Poor resource utilization")
        
        # Scalability
        scalability_scores = [r.metrics.linear_scalability_score for r in mode_results]
        avg_scalability = np.mean(scalability_scores)
        criteria_scores.append(avg_scalability)
        
        if avg_scalability > 0.7:
            assessment["strengths"].append("Good linear scalability")
        elif avg_scalability < 0.4:
            assessment["concerns"].append("Poor scalability characteristics")
        
        # Overall assessment
        assessment["overall_score"] = np.mean(criteria_scores) if criteria_scores else 0.0
        
        if assessment["overall_score"] > 0.8:
            assessment["readiness_level"] = "PRODUCTION_READY"
            assessment["recommendations"].append("Suitable for production deployment")
        elif assessment["overall_score"] > 0.6:
            assessment["readiness_level"] = "REQUIRES_OPTIMIZATION"
            assessment["recommendations"].append("Needs performance optimization before production")
        elif assessment["overall_score"] > 0.4:
            assessment["readiness_level"] = "DEVELOPMENT_ONLY"
            assessment["recommendations"].append("Use only for development and testing")
        else:
            assessment["readiness_level"] = "NOT_READY"
            assessment["recommendations"].append("Significant improvements needed")
        
        return assessment
    
    def _save_evaluation_results(self, analysis: Dict):
        """Save comprehensive evaluation results to single file"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create comprehensive results with embedded experiment data
        comprehensive_results = analysis.copy()
        
        # Embed experiment results to avoid separate file
        comprehensive_results["detailed_experiment_results"] = {
            k: asdict(v) for k, v in self.experiment_results.items()
        }
        
        # Save single comprehensive file
        results_file = self.base_results_dir / f"complete_evaluation_analysis_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        self.logger.info(f"💾 Comprehensive evaluation results saved: {results_file}")
    
    def _parse_scalability_metrics_string(self, metrics_str: str) -> Dict:
        """
        Parse scalability metrics from string representation
        
        Handles the case where scalability_metrics is stored as string instead of object
        """
        import re
        
        if not isinstance(metrics_str, str) or not metrics_str.startswith("ScalabilityMetrics"):
            return {}
        
        try:
            # Extract key=value pairs using regex
            pattern = r'(\w+)=([^,\)]+)'
            matches = re.findall(pattern, metrics_str)
            
            parsed = {}
            for key, value in matches:
                try:
                    # Clean up numpy float64 representations
                    if 'np.float64(' in value:
                        value = value.replace('np.float64(', '').replace(')', '')
                    
                    # Try to convert to float
                    if value.replace('.', '').replace('-', '').replace('e', '').replace('E', '').replace('+', '').isdigit():
                        parsed[key] = float(value)
                    else:
                        parsed[key] = value.strip("'\"")
                        
                except (ValueError, AttributeError):
                    parsed[key] = value.strip("'\"") if isinstance(value, str) else str(value)
            
            self.logger.debug(f"🔧 Parsed {len(parsed)} metrics from string representation")
            return parsed
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to parse scalability metrics string: {e}")
            return {}


# Utility functions for external use
def evaluate_volume_test_results(results_dir: str = "results/volume_tests") -> Dict:
    """Convenience function to evaluate all volume test results"""
    
    framework = EvaluationFramework()
    return framework.run_experiment_matrix_evaluation(results_dir)


def create_evaluation_summary(results_dir: str = "results/volume_tests") -> str:
    """Create a text summary of evaluation results"""
    
    framework = EvaluationFramework()
    analysis = framework.run_experiment_matrix_evaluation(results_dir)
    
    if not analysis:
        return "❌ No evaluation results available"
    
    # Generate summary
    summary_lines = [
        "📊 MULTI-MODE SCALABILITY EVALUATION SUMMARY",
        "=" * 60,
        "",
        f"📈 Evaluation Date: {analysis['evaluation_metadata']['evaluation_timestamp']}",
        f"🔬 Total Experiments: {analysis['evaluation_metadata']['total_experiments']}",
        f"📊 Volumes Tested: {', '.join(analysis['evaluation_metadata']['volumes_tested'])}",
        f"⚙️  Modes Tested: {', '.join(analysis['evaluation_metadata']['modes_tested'])}",
        "",
        "🏆 PERFORMANCE LEADERS:",
        "-" * 30
    ]
    
    # Add performance leaders
    leaders = analysis.get("comparative_analysis", {}).get("performance_leaders", {})
    for category, volume_data in leaders.items():
        summary_lines.append(f"\n{category.upper()}:")
        for volume, leader in volume_data.items():
            summary_lines.append(f"  {volume}: {leader['mode']}")
    
    # Add recommendations
    recommendations = analysis.get("performance_recommendations", {})
    if "deployment_scenarios" in recommendations:
        summary_lines.extend([
            "",
            "🎯 DEPLOYMENT RECOMMENDATIONS:",
            "-" * 35
        ])
        
        for scenario, rec in recommendations["deployment_scenarios"].items():
            summary_lines.append(f"\n{scenario}: {rec['recommended_mode']}")
            summary_lines.append(f"  - {rec['description']}")
    
    return "\n".join(summary_lines)


if __name__ == "__main__":
    # Run evaluation if called directly
    analysis = evaluate_volume_test_results()
    print(create_evaluation_summary())