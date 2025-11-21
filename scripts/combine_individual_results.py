#!/usr/bin/env python3
"""
Individual Results Combiner - Merge Individual Mode Test Results
Combines results from separate individual mode runs into a unified analysis for Phase 5 validation

Usage:
    # Run individual mode tests first:
    python scripts/test_volume_scalability.py --volume large --mode spark_gbt
    python scripts/test_volume_scalability.py --volume large --mode spark_rf  
    python scripts/test_volume_scalability.py --volume large --mode hybrid_sklearn
    
    # Then combine results:
    python scripts/combine_individual_results.py \
        --result1 results/volume_tests/individual/individual_spark_gbt_large_20241120_120000.json \
        --result2 results/volume_tests/individual/individual_spark_rf_large_20241120_120100.json \
        --result3 results/volume_tests/individual/individual_hybrid_sklearn_large_20241120_120200.json \
        --output results/volume_tests/large_volume_test_combined.json
    
    # Then run Phase 5 validation:
    python scripts/test_phase5_validation.py
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

# Add src to path
pipeline_root = Path(__file__).parent.parent
sys.path.insert(0, str(pipeline_root))

from src.utils import Logger


class IndividualResultsCombiner:
    """Combines individual mode test results into unified analysis"""
    
    def __init__(self):
        self.logger = Logger().get_logger()
        
    def combine_results(self, result_files: List[str], output_path: str = None) -> str:
        """
        Combine individual mode test results into a unified analysis
        
        Args:
            result_files: List of paths to individual result JSON files
            output_path: Optional output path. If None, auto-generates based on volume
        
        Returns:
            Path to the combined result file
        """
        
        if len(result_files) < 1:
            raise ValueError("At least 1 result file is required")
        
        self.logger.info(f"🔄 Combining {len(result_files)} individual test results...")
        
        # Load and validate all result files
        individual_results = []
        volume = None
        volume_spec = None
        modes_tested = []
        mode_results = {}
        
        total_session_time = 0.0
        earliest_timestamp = None
        latest_timestamp = None
        
        for i, file_path in enumerate(result_files):
            self.logger.info(f"📄 Loading result {i+1}: {file_path}")
            
            if not os.path.exists(file_path):
                self.logger.error(f"❌ File not found: {file_path}")
                continue
            
            try:
                with open(file_path, 'r') as f:
                    result_data = json.load(f)
                
                # Validate this is an individual test result
                if not result_data.get("individual_mode_test", False):
                    self.logger.warning(f"⚠️ File {file_path} is not an individual mode test result")
                
                # Extract basic info
                file_volume = result_data.get("volume")
                file_volume_spec = result_data.get("volume_spec")
                file_modes = result_data.get("modes_tested", [])
                file_mode_results = result_data.get("mode_results", {})
                
                # Validate consistency
                if volume is None:
                    volume = file_volume
                    volume_spec = file_volume_spec
                elif volume != file_volume:
                    self.logger.warning(f"⚠️ Volume mismatch: {volume} vs {file_volume}")
                
                # Add modes and results
                for mode in file_modes:
                    if mode not in modes_tested:
                        modes_tested.append(mode)
                    
                    if mode in file_mode_results:
                        if mode in mode_results:
                            self.logger.warning(f"⚠️ Duplicate mode {mode} - using latest result")
                        mode_results[mode] = file_mode_results[mode]
                
                # Track timing
                session_time = result_data.get("total_session_time", 0)
                total_session_time += session_time
                
                # Track timestamps for overall session span
                test_timestamp = result_data.get("test_session_timestamp")
                if test_timestamp:
                    try:
                        ts = datetime.fromisoformat(test_timestamp.replace('Z', '+00:00'))
                        if earliest_timestamp is None or ts < earliest_timestamp:
                            earliest_timestamp = ts
                        if latest_timestamp is None or ts > latest_timestamp:
                            latest_timestamp = ts
                    except Exception as e:
                        self.logger.warning(f"⚠️ Could not parse timestamp {test_timestamp}: {e}")
                
                individual_results.append(result_data)
                self.logger.info(f"✅ Loaded {len(file_modes)} mode(s): {file_modes}")
                
            except Exception as e:
                self.logger.error(f"❌ Error loading {file_path}: {e}")
                # Include error in combined result
                error_mode = f"error_file_{i+1}"
                modes_tested.append(error_mode)
                mode_results[error_mode] = {
                    "mode": error_mode,
                    "volume": volume or "unknown",
                    "error": str(e),
                    "test_timestamp": datetime.now().isoformat(),
                    "source_file": file_path
                }
        
        if not mode_results:
            raise ValueError("No valid results found in any of the provided files")
        
        # Create comparative analysis
        comparative_analysis = self._create_combined_comparative_analysis(mode_results)
        
        # Calculate session span if we have timestamps
        estimated_session_span = total_session_time  # Default to sum
        if earliest_timestamp and latest_timestamp:
            actual_span = (latest_timestamp - earliest_timestamp).total_seconds()
            estimated_session_span = actual_span
        
        # Create combined result with same structure as multi-mode test
        combined_result = {
            "volume": volume,
            "volume_spec": volume_spec,
            "modes_tested": modes_tested,
            "test_session_timestamp": (latest_timestamp or datetime.now()).isoformat(),
            "total_session_time": total_session_time,
            "estimated_session_span_seconds": estimated_session_span,
            "mode_results": mode_results,
            "comparative_analysis": comparative_analysis,
            "combination_metadata": {
                "combined_from_individual_tests": True,
                "source_files": result_files,
                "combination_timestamp": datetime.now().isoformat(),
                "individual_test_count": len(individual_results),
                "successful_modes": len([m for m in modes_tested if not m.startswith("error_")]),
                "failed_modes": len([m for m in modes_tested if m.startswith("error_")])
            }
        }
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"results/volume_tests/{volume}_volume_test_combined_{timestamp}.json"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save combined result
        with open(output_path, 'w') as f:
            json.dump(combined_result, f, indent=2, default=str)
        
        self.logger.info(f"✅ Combined result saved: {output_path}")
        self.logger.info(f"📊 Combined {len(modes_tested)} modes for {volume} volume")
        self.logger.info(f"⏱️ Total session time: {total_session_time:.2f}s")
        self.logger.info(f"📈 Session efficiency: {estimated_session_span/total_session_time*100:.1f}%")
        
        return output_path
    
    def _create_combined_comparative_analysis(self, mode_results: Dict) -> Dict:
        """Create comparative analysis from combined mode results"""
        
        analysis = {
            "fastest_mode": None,
            "most_memory_efficient": None,
            "best_cost_efficiency": None,
            "mode_rankings": [],
            "performance_comparison": {}
        }
        
        try:
            # Filter out error modes for analysis
            valid_modes = {k: v for k, v in mode_results.items() 
                          if not k.startswith("error_") and "error" not in v}
            
            if not valid_modes:
                analysis["note"] = "No valid modes for comparative analysis"
                return analysis
            
            # Analyze performance metrics
            mode_performance = {}
            
            for mode, result in valid_modes.items():
                training_result = result.get("training_result", {})
                performance_summary = result.get("performance_summary", {})
                
                # Extract key metrics
                training_time = performance_summary.get("training_time", training_result.get("training_time", float('inf')))
                memory_efficiency = performance_summary.get("memory_efficiency", 0)
                cost_efficiency = performance_summary.get("cost_efficiency", float('inf'))
                
                # Extract test metrics for overall performance score
                test_metrics = training_result.get("metrics", {}).get("test_metrics", {})
                r2_score = test_metrics.get("r2", 0)
                
                mode_performance[mode] = {
                    "training_time": training_time,
                    "memory_efficiency": memory_efficiency,
                    "cost_efficiency": cost_efficiency,
                    "r2_score": r2_score,
                    "overall_score": r2_score * 0.4 + memory_efficiency * 0.3 + (1/(training_time+1)) * 0.3
                }
            
            # Determine best modes
            if mode_performance:
                # Fastest mode (lowest training time)
                fastest = min(mode_performance.items(), key=lambda x: x[1]["training_time"])
                analysis["fastest_mode"] = fastest[0]
                
                # Most memory efficient (highest memory efficiency score)
                memory_efficient = max(mode_performance.items(), key=lambda x: x[1]["memory_efficiency"])
                analysis["most_memory_efficient"] = memory_efficient[0]
                
                # Best cost efficiency (lowest cost)
                cost_efficient = min(mode_performance.items(), key=lambda x: x[1]["cost_efficiency"])
                analysis["best_cost_efficiency"] = cost_efficient[0]
                
                # Overall ranking by combined score
                ranked_modes = sorted(mode_performance.items(), key=lambda x: x[1]["overall_score"], reverse=True)
                analysis["mode_rankings"] = [
                    {"mode": mode, "score": perf["overall_score"], "rank": i+1}
                    for i, (mode, perf) in enumerate(ranked_modes)
                ]
                
                # Performance comparison
                analysis["performance_comparison"] = {
                    mode: {
                        "training_time_seconds": perf["training_time"],
                        "memory_efficiency_score": perf["memory_efficiency"],
                        "cost_efficiency_score": perf["cost_efficiency"],
                        "model_accuracy_r2": perf["r2_score"],
                        "overall_score": perf["overall_score"]
                    }
                    for mode, perf in mode_performance.items()
                }
        
        except Exception as e:
            self.logger.warning(f"⚠️ Error creating comparative analysis: {e}")
            analysis["note"] = f"Comparative analysis error: {str(e)}"
        
        return analysis


def main():
    """Main function for combining individual results"""
    
    parser = argparse.ArgumentParser(
        description="Combine individual mode test results into unified analysis"
    )
    
    parser.add_argument(
        '--result1',
        required=True,
        help='Path to first individual test result JSON file'
    )
    
    parser.add_argument(
        '--result2',
        help='Path to second individual test result JSON file'
    )
    
    parser.add_argument(
        '--result3',
        help='Path to third individual test result JSON file'
    )
    
    parser.add_argument(
        '--output',
        help='Output path for combined result (optional, auto-generated if not provided)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Collect all provided result files
    result_files = [args.result1]
    if args.result2:
        result_files.append(args.result2)
    if args.result3:
        result_files.append(args.result3)
    
    # Set up logging
    logger = Logger().get_logger()
    
    if args.verbose:
        logger.info(f"🔍 Input files: {result_files}")
        logger.info(f"📁 Output: {args.output or 'auto-generated'}")
    
    # Create combiner and process
    combiner = IndividualResultsCombiner()
    
    try:
        output_path = combiner.combine_results(result_files, args.output)
        
        print(f"✅ COMBINATION SUCCESSFUL")
        print(f"📁 Combined analysis saved: {output_path}")
        print(f"🚀 Next step: python scripts/test_phase5_validation.py")
        
    except Exception as e:
        print(f"❌ COMBINATION FAILED: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()