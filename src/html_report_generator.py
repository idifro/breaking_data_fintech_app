#!/usr/bin/env python3
"""
Comprehensive HTML Report Generator for Multi-Mode Scalability Analysis
Creates detailed HTML reports with embedded plots and analysis

Part of Data Engineering at Scale project - Phase 5 implementation
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import base64
from jinja2 import Template

from src.utils import Logger
from src.evaluation_framework import EvaluationFramework


class HTMLReportGenerator:
    """
    Generate comprehensive HTML reports for scalability analysis
    Includes embedded plots, detailed metrics, and production recommendations
    """
    
    def __init__(self, output_dir: str = "reports/html"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = Logger().get_logger()
        
        self.logger.info("📄 HTML Report Generator initialized")
    
    def generate_comprehensive_report(self, evaluation_results: Dict, 
                                    plots_directory: str = None) -> str:
        """Generate comprehensive HTML report with all analysis results"""
        
        self.logger.info("📄 Generating comprehensive HTML report...")
        
        try:
            # Prepare report data
            report_data = self._prepare_report_data(evaluation_results)
            
            # Embed plot images if available
            if plots_directory:
                embedded_plots = self._embed_plot_images(plots_directory)
                report_data["embedded_plots"] = embedded_plots
            else:
                report_data["embedded_plots"] = {}
            
            # Generate HTML content
            html_content = self._generate_html_content(report_data)
            
            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"scalability_analysis_report_{timestamp}.html"
            report_path = self.output_dir / report_filename
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"✅ Comprehensive report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"❌ HTML report generation failed: {e}")
            return ""
    
    def _generate_report_with_custom_name(self, evaluation_results: Dict, 
                                        plots_directory: str = None,
                                        custom_filename: str = None) -> str:
        """Generate report with custom filename to avoid conflicts"""
        
        self.logger.info("📄 Generating validation HTML report...")
        
        try:
            # Prepare report data
            report_data = self._prepare_report_data(evaluation_results)
            
            # Embed plot images if available
            if plots_directory:
                embedded_plots = self._embed_plot_images(plots_directory)
                report_data["embedded_plots"] = embedded_plots
            else:
                report_data["embedded_plots"] = {}
            
            # Generate HTML content
            html_content = self._generate_html_content(report_data)
            
            # Save report with custom filename
            report_filename = custom_filename or f"scalability_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            report_path = self.output_dir / report_filename
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"✅ Validation report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"❌ Validation report generation failed: {e}")
            return ""
    
    def _prepare_report_data(self, evaluation_results: Dict) -> Dict:
        """Prepare data structure for HTML template"""
        
        try:
            # Extract key metrics for report
            report_data = {
                "generation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "project_title": "Multi-Mode Scalability Analysis Report",
                "evaluation_summary": evaluation_results.get("evaluation_summary", {}),
                "experiment_matrix": evaluation_results.get("experiment_matrix", {}),
                "scalability_analysis": evaluation_results.get("scalability_analysis", {}),
                "production_recommendations": evaluation_results.get("production_recommendations", {}),
                "performance_metrics": self._extract_performance_metrics(evaluation_results),
                "mode_comparisons": self._extract_mode_comparisons(evaluation_results)
            }
            
            return report_data
            
        except Exception as e:
            self.logger.error(f"❌ Report data preparation failed: {e}")
            return {}
    
    def _extract_performance_metrics(self, evaluation_results: Dict) -> Dict:
        """Extract and organize performance metrics for display"""
        
        metrics = {}
        
        try:
            experiment_matrix = evaluation_results.get("experiment_matrix", {})
            results = experiment_matrix.get("results", {})
            
            for mode in ['spark_gbt', 'spark_rf', 'hybrid_sklearn']:
                if mode not in results:
                    continue
                
                metrics[mode] = {
                    "mode_name": mode.replace('_', ' ').title(),
                    "volumes": {}
                }
                
                for volume in ['small', 'medium', 'large', 'full']:
                    if (volume in results[mode] and 
                        results[mode][volume]["status"] == "SUCCESS"):
                        
                        volume_metrics = results[mode][volume]["metrics"]
                        if volume_metrics:
                            metrics[mode]["volumes"][volume] = {
                                "training_time": volume_metrics.get("training_time_seconds", 0),
                                "memory_usage": volume_metrics.get("peak_memory_usage_mb", 0),
                                "throughput": volume_metrics.get("throughput_records_per_second", 0),
                                "accuracy": volume_metrics.get("r2_score", 0),
                                "cost_efficiency": volume_metrics.get("cost_efficiency_score", 0),
                                "scalability_score": volume_metrics.get("linear_scalability_score", 0)
                            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Performance metrics extraction failed: {e}")
            return {}
    
    def _extract_mode_comparisons(self, evaluation_results: Dict) -> Dict:
        """Extract mode comparison data for tabular display"""
        
        comparisons = {}
        
        try:
            scalability_analysis = evaluation_results.get("scalability_analysis", {})
            
            # Best performers by category
            comparisons["best_performers"] = {
                "fastest_training": "N/A",
                "most_memory_efficient": "N/A", 
                "best_scalability": "N/A",
                "highest_accuracy": "N/A"
            }
            
            # Try to extract best performers
            if "mode_rankings" in scalability_analysis:
                rankings = scalability_analysis["mode_rankings"]
                
                # Find best performers (assuming rankings are sorted)
                for category, modes in rankings.items():
                    if modes:
                        if "speed" in category.lower() or "time" in category.lower():
                            comparisons["best_performers"]["fastest_training"] = modes[0].replace('_', ' ').title()
                        elif "memory" in category.lower():
                            comparisons["best_performers"]["most_memory_efficient"] = modes[0].replace('_', ' ').title()
                        elif "scalability" in category.lower():
                            comparisons["best_performers"]["best_scalability"] = modes[0].replace('_', ' ').title()
                        elif "accuracy" in category.lower():
                            comparisons["best_performers"]["highest_accuracy"] = modes[0].replace('_', ' ').title()
            
            return comparisons
            
        except Exception as e:
            self.logger.error(f"❌ Mode comparison extraction failed: {e}")
            return {}
    
    def _embed_plot_images(self, plots_directory: str) -> Dict[str, str]:
        """Embed plot images as base64 encoded data URLs"""
        
        embedded_plots = {}
        
        try:
            plots_path = Path(plots_directory)
            if not plots_path.exists():
                self.logger.warning(f"⚠️ Plots directory not found: {plots_directory}")
                return embedded_plots
            
            # Expected plot files
            plot_files = {
                "training_time_scalability": "training_time_scalability.png",
                "memory_usage_comparison": "memory_usage_comparison.png",
                "resource_utilization_heatmap": "resource_utilization_heatmap.png",
                "performance_accuracy_scatter": "performance_accuracy_scatter.png",
                "mode_performance_radar": "mode_performance_radar.png"
            }
            
            for plot_name, filename in plot_files.items():
                plot_path = plots_path / filename
                if plot_path.exists():
                    try:
                        with open(plot_path, 'rb') as f:
                            image_data = f.read()
                            encoded_image = base64.b64encode(image_data).decode('utf-8')
                            embedded_plots[plot_name] = f"data:image/png;base64,{encoded_image}"
                        
                        self.logger.debug(f"🖼️ Embedded plot: {plot_name}")
                        
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to embed plot {plot_name}: {e}")
                else:
                    self.logger.warning(f"⚠️ Plot file not found: {plot_path}")
            
            self.logger.info(f"🖼️ Embedded {len(embedded_plots)} plot images")
            return embedded_plots
            
        except Exception as e:
            self.logger.error(f"❌ Plot embedding failed: {e}")
            return {}
    
    def _generate_html_content(self, report_data: Dict) -> str:
        """Generate HTML content using template"""
        
        # HTML template with embedded CSS and JavaScript
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ report_data.project_title }}</title>
    <style>
        /* Modern CSS styling */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem 0;
            text-align: center;
            margin-bottom: 2rem;
            border-radius: 10px;
        }

        header h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }

        header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }

        .section {
            background: white;
            margin-bottom: 2rem;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .section-title {
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 0.5rem;
            margin-bottom: 1.5rem;
            font-size: 1.5rem;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }

        .metric-card {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
        }

        .metric-label {
            color: #666;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .plot-container {
            text-align: center;
            margin: 2rem 0;
        }

        .plot-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        .plot-title {
            font-weight: bold;
            margin-bottom: 1rem;
            color: #333;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }

        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }

        th {
            background-color: #667eea;
            color: white;
            font-weight: 600;
        }

        tr:hover {
            background-color: #f5f5f5;
        }

        .status-success {
            color: #28a745;
            font-weight: bold;
        }

        .status-warning {
            color: #ffc107;
            font-weight: bold;
        }

        .status-error {
            color: #dc3545;
            font-weight: bold;
        }

        .recommendation {
            background: #e8f5e8;
            border-left: 4px solid #28a745;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 4px;
        }

        .warning {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 4px;
        }

        .tabs {
            display: flex;
            border-bottom: 2px solid #ddd;
            margin-bottom: 1rem;
        }

        .tab {
            padding: 10px 20px;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.3s;
        }

        .tab.active {
            border-bottom-color: #667eea;
            color: #667eea;
            font-weight: bold;
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        .footer {
            text-align: center;
            padding: 2rem;
            color: #666;
            border-top: 1px solid #ddd;
            margin-top: 3rem;
        }

        @media (max-width: 768px) {
            .metrics-grid {
                grid-template-columns: 1fr;
            }
            
            header h1 {
                font-size: 2rem;
            }
            
            .container {
                padding: 10px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{{ report_data.project_title }}</h1>
            <p>Generated on {{ report_data.generation_timestamp }}</p>
        </header>

        <!-- Executive Summary -->
        <div class="section">
            <h2 class="section-title">📊 Executive Summary</h2>
            
            {% if report_data.evaluation_summary %}
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{{ report_data.evaluation_summary.get('total_experiments', 0) }}</div>
                    <div class="metric-label">Total Experiments</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ report_data.evaluation_summary.get('successful_experiments', 0) }}</div>
                    <div class="metric-label">Successful Runs</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ "%.1f"|format(report_data.evaluation_summary.get('total_execution_time_hours', 0)) }}h</div>
                    <div class="metric-label">Total Execution Time</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ "%.1f"|format(report_data.evaluation_summary.get('average_accuracy', 0) * 100) }}%</div>
                    <div class="metric-label">Average R² Score</div>
                </div>
            </div>
            {% endif %}

            {% if report_data.mode_comparisons and report_data.mode_comparisons.get('best_performers') %}
            <h3>🏆 Best Performers by Category</h3>
            <table>
                <tr>
                    <th>Category</th>
                    <th>Winner</th>
                </tr>
                {% for category, winner in report_data.mode_comparisons.best_performers.items() %}
                <tr>
                    <td>{{ category.replace('_', ' ').title() }}</td>
                    <td class="status-success">{{ winner }}</td>
                </tr>
                {% endfor %}
            </table>
            {% endif %}
        </div>

        <!-- Performance Visualizations -->
        <div class="section">
            <h2 class="section-title">📈 Performance Visualizations</h2>
            
            <div class="tabs">
                <div class="tab active" onclick="showTab('scalability')">Scalability</div>
                <div class="tab" onclick="showTab('resource')">Resource Usage</div>
                <div class="tab" onclick="showTab('comparison')">Mode Comparison</div>
            </div>

            <div id="scalability" class="tab-content active">
                {% if report_data.embedded_plots.get('training_time_scalability') %}
                <div class="plot-container">
                    <div class="plot-title">Training Time Scalability Analysis</div>
                    <img src="{{ report_data.embedded_plots.training_time_scalability }}" alt="Training Time Scalability">
                </div>
                {% endif %}

                {% if report_data.embedded_plots.get('performance_accuracy_scatter') %}
                <div class="plot-container">
                    <div class="plot-title">Performance vs Accuracy Trade-off</div>
                    <img src="{{ report_data.embedded_plots.performance_accuracy_scatter }}" alt="Performance vs Accuracy">
                </div>
                {% endif %}
            </div>

            <div id="resource" class="tab-content">
                {% if report_data.embedded_plots.get('memory_usage_comparison') %}
                <div class="plot-container">
                    <div class="plot-title">Memory Usage Comparison</div>
                    <img src="{{ report_data.embedded_plots.memory_usage_comparison }}" alt="Memory Usage Comparison">
                </div>
                {% endif %}

                {% if report_data.embedded_plots.get('resource_utilization_heatmap') %}
                <div class="plot-container">
                    <div class="plot-title">Resource Utilization Heatmap</div>
                    <img src="{{ report_data.embedded_plots.resource_utilization_heatmap }}" alt="Resource Utilization Heatmap">
                </div>
                {% endif %}
            </div>

            <div id="comparison" class="tab-content">
                {% if report_data.embedded_plots.get('mode_performance_radar') %}
                <div class="plot-container">
                    <div class="plot-title">Mode Performance Radar Chart</div>
                    <img src="{{ report_data.embedded_plots.mode_performance_radar }}" alt="Mode Performance Radar">
                </div>
                {% endif %}

            </div>
        </div>

        <!-- Detailed Performance Metrics -->
        <div class="section">
            <h2 class="section-title">⚡ Detailed Performance Metrics</h2>
            
            {% for mode_key, mode_data in report_data.performance_metrics.items() %}
            <h3>{{ mode_data.mode_name }}</h3>
            <table>
                <tr>
                    <th>Volume</th>
                    <th>Training Time (s)</th>
                    <th>Memory Usage (MB)</th>
                    <th>Throughput (rec/s)</th>
                    <th>Accuracy (R²)</th>
                    <th>Cost Efficiency</th>
                </tr>
                {% for volume, metrics in mode_data.volumes.items() %}
                <tr>
                    <td>{{ volume.title() }}</td>
                    <td>{{ "%.1f"|format(metrics.training_time) }}</td>
                    <td>{{ "%.1f"|format(metrics.memory_usage) }}</td>
                    <td>{{ "%.1f"|format(metrics.throughput) }}</td>
                    <td>{{ "%.3f"|format(metrics.accuracy) }}</td>
                    <td>{{ "%.3f"|format(metrics.cost_efficiency) }}</td>
                </tr>
                {% endfor %}
            </table>
            {% endfor %}
        </div>

        <!-- Production Recommendations -->
        <div class="section">
            <h2 class="section-title">🚀 Production Recommendations</h2>
            
            {% if report_data.production_recommendations %}
                {% if report_data.production_recommendations.get('recommended_mode') %}
                <div class="recommendation">
                    <h3>🎯 Recommended Mode</h3>
                    <p><strong>{{ report_data.production_recommendations.recommended_mode.replace('_', ' ').title() }}</strong></p>
                    {% if report_data.production_recommendations.get('recommendation_reasoning') %}
                    <p>{{ report_data.production_recommendations.recommendation_reasoning }}</p>
                    {% endif %}
                </div>
                {% endif %}

                {% if report_data.production_recommendations.get('scaling_guidelines') %}
                <h3>📏 Scaling Guidelines</h3>
                <ul>
                    {% for guideline in report_data.production_recommendations.scaling_guidelines %}
                    <li>{{ guideline }}</li>
                    {% endfor %}
                </ul>
                {% endif %}

                {% if report_data.production_recommendations.get('resource_requirements') %}
                <h3>💻 Resource Requirements</h3>
                <table>
                    <tr>
                        <th>Volume</th>
                        <th>Recommended Memory</th>
                        <th>Recommended Cores</th>
                        <th>Expected Time</th>
                    </tr>
                    {% for volume, reqs in report_data.production_recommendations.resource_requirements.items() %}
                    <tr>
                        <td>{{ volume.title() }}</td>
                        <td>{{ reqs.get('memory_gb', 'N/A') }}GB</td>
                        <td>{{ reqs.get('cpu_cores', 'N/A') }}</td>
                        <td>{{ reqs.get('estimated_time_minutes', 'N/A') }}min</td>
                    </tr>
                    {% endfor %}
                </table>
                {% endif %}
            {% else %}
            <p>Production recommendations will be available after completing evaluation across all volumes.</p>
            {% endif %}
        </div>

        <!-- Technical Details -->
        <div class="section">
            <h2 class="section-title">🔧 Technical Details</h2>
            
            <h3>Experiment Configuration</h3>
            <ul>
                <li><strong>Spark Version:</strong> 3.4.1</li>
                <li><strong>Python Version:</strong> 3.9+</li>
                <li><strong>Test Volumes:</strong> Small (5 stocks), Medium (15 stocks), Large (25 stocks), Full (29 stocks)</li>
                <li><strong>Training Modes:</strong> Spark GBT, Spark Random Forest, Hybrid Sklearn</li>
                <li><strong>Evaluation Metrics:</strong> Training Time, Memory Usage, Throughput, Model Accuracy (R²)</li>
                <li><strong>Cost Calculation:</strong> Memory (MB) × Time (seconds)</li>
            </ul>

            <h3>Data Pipeline Architecture</h3>
            <ul>
                <li><strong>Data Processing:</strong> Apache Spark with Delta Lake storage format</li>
                <li><strong>Feature Engineering:</strong> Technical indicators, rolling statistics, lag features</li>
                <li><strong>Model Training:</strong> MLflow experiment tracking with distributed training</li>
                <li><strong>Evaluation Framework:</strong> Comprehensive scalability and performance analysis</li>
                <li><strong>Monitoring:</strong> Real-time resource utilization tracking</li>
            </ul>
        </div>

        <div class="footer">
            <p>Data Engineering at Scale - Multi-Mode Scalability Analysis</p>
            <p>Report generated automatically by the ML Pipeline Framework</p>
        </div>
    </div>

    <script>
        function showTab(tabName) {
            // Hide all tab contents
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => content.classList.remove('active'));
            
            // Remove active class from all tabs
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => tab.classList.remove('active'));
            
            // Show selected tab content
            document.getElementById(tabName).classList.add('active');
            
            // Mark selected tab as active
            event.target.classList.add('active');
        }
    </script>
</body>
</html>
        """
        
        try:
            # Create template and render
            template = Template(html_template)
            html_content = template.render(report_data=report_data)
            
            self.logger.info("📄 HTML content generated successfully")
            return html_content
            
        except Exception as e:
            self.logger.error(f"❌ HTML template rendering failed: {e}")
            return f"<html><body><h1>Report Generation Error</h1><p>{str(e)}</p></body></html>"


# Utility function to generate complete report
def generate_complete_html_report(results_dir: str = "results/volume_tests",
                                plots_dir: str = "plots/advanced_analysis") -> str:
    """Generate complete HTML report with visualizations"""
    
    logger = Logger().get_logger()
    
    try:
        # Load evaluation results
        from src.evaluation_framework import evaluate_volume_test_results
        evaluation_results = evaluate_volume_test_results(results_dir)
        
        if not evaluation_results:
            logger.error("❌ No evaluation results available for report")
            return ""
        
        # Create report generator
        report_generator = HTMLReportGenerator()
        
        # Generate comprehensive report
        report_path = report_generator.generate_comprehensive_report(
            evaluation_results, plots_dir
        )
        
        logger.info(f"📄 Complete HTML report generated: {report_path}")
        return report_path
        
    except Exception as e:
        logger.error(f"❌ Complete report generation failed: {e}")
        return ""


if __name__ == "__main__":
    # Generate report if called directly
    report_path = generate_complete_html_report()
    if report_path:
        print(f"✅ HTML report generated: {report_path}")
    else:
        print("❌ Report generation failed")