"""
Visualization Module for Stock Price Prediction
Creates plots for feature importance, predictions, and model performance
"""

from typing import List, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from src.utils import Config, PathManager


class Visualizer:
    """Visualization class for model evaluation and analysis"""
    
    def __init__(self, config: Config):
        self.config = config
        self.path_manager = PathManager()
        self.plots_dir = self.path_manager.plots_dir
        
        # Set style
        plt.style.use(config.visualization.get('style', 'seaborn-v0_8'))
        sns.set_palette("husl")
        
        # Create plots directory
        self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n: int = 20) -> None:
        """Plot feature importance"""
        
        print("📊 Creating feature importance plot...")
        
        # Get top N features
        top_features = importance_df.head(top_n)
        
        # Create matplotlib plot
        fig, ax = plt.subplots(figsize=self.config.visualization.get('figure_size', [12, 8]))
        
        bars = ax.barh(range(len(top_features)), top_features['importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {top_n} Feature Importance - GBT Model')
        
        # Color bars
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add value labels
        for i, v in enumerate(top_features['importance']):
            ax.text(v + 0.001, i, f'{v:.4f}', va='center', ha='left')
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.plots_dir / "feature_importance.png"
        plt.savefig(save_path, dpi=self.config.visualization.get('dpi', 300), 
                    bbox_inches='tight')
        plt.close()
        
        # Create interactive plotly version
        fig_plotly = px.bar(
            top_features.iloc[::-1],  # Reverse for better display
            x='importance',
            y='feature',
            orientation='h',
            title=f'Top {top_n} Feature Importance - GBT Model',
            labels={'importance': 'Feature Importance', 'feature': 'Features'}
        )
        
        fig_plotly.update_layout(
            height=max(400, len(top_features) * 25),
            showlegend=False
        )
        
        # Save interactive plot
        save_path_html = self.plots_dir / "feature_importance_interactive.html"
        fig_plotly.write_html(str(save_path_html))
        
        print(f"✅ Feature importance plots saved to: {self.plots_dir}")
    
    def plot_predictions_vs_actual(self, test_results: List, sample_size: int = 1000) -> None:
        """Plot predictions vs actual values"""
        
        print("📊 Creating predictions vs actual plot...")
        
        # Convert to DataFrame
        results_df = pd.DataFrame([
            {
                'stock_symbol': row.stock_symbol,
                'actual': row.target,
                'predicted': row.prediction,
                'date': row.Date if hasattr(row, 'Date') else None
            }
            for row in test_results
        ])
        
        # Sample data if too large
        if len(results_df) > sample_size:
            results_df = results_df.sample(n=sample_size, random_state=42)
        
        # Create scatter plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Overall scatter plot
        ax1 = axes[0, 0]
        scatter = ax1.scatter(results_df['actual'], results_df['predicted'], 
                             alpha=0.6, c=results_df['stock_symbol'].astype('category').cat.codes, 
                             cmap='tab10')
        
        # Perfect prediction line
        min_val = min(results_df['actual'].min(), results_df['predicted'].min())
        max_val = max(results_df['actual'].max(), results_df['predicted'].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        ax1.set_xlabel('Actual Returns')
        ax1.set_ylabel('Predicted Returns')
        ax1.set_title('Predictions vs Actual - All Stocks')
        ax1.grid(True, alpha=0.3)
        
        # Add R² score
        from sklearn.metrics import r2_score
        r2 = r2_score(results_df['actual'], results_df['predicted'])
        ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Residuals plot
        ax2 = axes[0, 1]
        residuals = results_df['actual'] - results_df['predicted']
        ax2.scatter(results_df['predicted'], residuals, alpha=0.6)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax2.set_xlabel('Predicted Returns')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residuals vs Predicted')
        ax2.grid(True, alpha=0.3)
        
        # Distribution of predictions and actual
        ax3 = axes[1, 0]
        ax3.hist(results_df['actual'], bins=50, alpha=0.7, label='Actual', density=True)
        ax3.hist(results_df['predicted'], bins=50, alpha=0.7, label='Predicted', density=True)
        ax3.set_xlabel('Returns')
        ax3.set_ylabel('Density')
        ax3.set_title('Distribution of Returns')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Error distribution
        ax4 = axes[1, 1]
        errors = abs(results_df['actual'] - results_df['predicted'])
        ax4.hist(errors, bins=50, alpha=0.7, color='orange')
        ax4.set_xlabel('Absolute Error')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Absolute Errors')
        ax4.grid(True, alpha=0.3)
        
        # Add mean error
        mean_error = errors.mean()
        ax4.axvline(x=mean_error, color='r', linestyle='--', 
                   label=f'Mean Error: {mean_error:.6f}')
        ax4.legend()
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.plots_dir / "predictions_vs_actual.png"
        plt.savefig(save_path, dpi=self.config.visualization.get('dpi', 300), 
                    bbox_inches='tight')
        plt.close()
        
        print(f"✅ Predictions vs actual plot saved to: {save_path}")
    
    def plot_per_stock_performance(self, stock_metrics: Dict[str, Dict[str, float]]) -> None:
        """Plot performance metrics per stock"""
        
        print("📊 Creating per-stock performance plot...")
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame(stock_metrics).T
        metrics_df.reset_index(inplace=True)
        metrics_df.rename(columns={'index': 'stock'}, inplace=True)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # MAE by stock
        ax1 = axes[0, 0]
        bars1 = ax1.bar(metrics_df['stock'], metrics_df['mae'])
        ax1.set_xlabel('Stock Symbol')
        ax1.set_ylabel('Mean Absolute Error')
        ax1.set_title('MAE by Stock')
        ax1.tick_params(axis='x', rotation=45)
        
        # Color bars by performance
        colors1 = plt.cm.RdYlBu_r(metrics_df['mae'] / metrics_df['mae'].max())
        for bar, color in zip(bars1, colors1):
            bar.set_color(color)
        
        # MSE by stock
        ax2 = axes[0, 1]
        bars2 = ax2.bar(metrics_df['stock'], metrics_df['mse'])
        ax2.set_xlabel('Stock Symbol')
        ax2.set_ylabel('Mean Squared Error')
        ax2.set_title('MSE by Stock')
        ax2.tick_params(axis='x', rotation=45)
        
        colors2 = plt.cm.RdYlBu_r(metrics_df['mse'] / metrics_df['mse'].max())
        for bar, color in zip(bars2, colors2):
            bar.set_color(color)
        
        # R² by stock
        ax3 = axes[1, 0]
        bars3 = ax3.bar(metrics_df['stock'], metrics_df['r2'])
        ax3.set_xlabel('Stock Symbol')
        ax3.set_ylabel('R² Score')
        ax3.set_title('R² Score by Stock')
        ax3.tick_params(axis='x', rotation=45)
        
        colors3 = plt.cm.RdYlBu(metrics_df['r2'] / metrics_df['r2'].max())
        for bar, color in zip(bars3, colors3):
            bar.set_color(color)
        
        # Sample count by stock
        ax4 = axes[1, 1]
        bars4 = ax4.bar(metrics_df['stock'], metrics_df['sample_count'])
        ax4.set_xlabel('Stock Symbol')
        ax4.set_ylabel('Sample Count')
        ax4.set_title('Test Samples by Stock')
        ax4.tick_params(axis='x', rotation=45)
        
        for bar in bars4:
            bar.set_color('steelblue')
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.plots_dir / "per_stock_performance.png"
        plt.savefig(save_path, dpi=self.config.visualization.get('dpi', 300), 
                    bbox_inches='tight')
        plt.close()
        
        print(f"✅ Per-stock performance plot saved to: {save_path}")
    
    def plot_residuals(self, test_results: List) -> None:
        """Plot detailed residual analysis"""
        
        print("📊 Creating residuals analysis plot...")
        
        # Convert to DataFrame
        results_df = pd.DataFrame([
            {
                'stock_symbol': row.stock_symbol,
                'actual': row.target,
                'predicted': row.prediction,
                'residual': row.target - row.prediction,
                'date': row.Date if hasattr(row, 'Date') else None
            }
            for row in test_results
        ])
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Residuals vs predicted
        ax1 = axes[0, 0]
        ax1.scatter(results_df['predicted'], results_df['residual'], alpha=0.6)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted')
        ax1.grid(True, alpha=0.3)
        
        # Residuals histogram
        ax2 = axes[0, 1]
        ax2.hist(results_df['residual'], bins=50, alpha=0.7, density=True)
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Density')
        ax2.set_title('Distribution of Residuals')
        ax2.grid(True, alpha=0.3)
        
        # Add normal distribution overlay
        mean_residual = results_df['residual'].mean()
        std_residual = results_df['residual'].std()
        x_norm = np.linspace(results_df['residual'].min(), results_df['residual'].max(), 100)
        y_norm = (1/(std_residual * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - mean_residual) / std_residual) ** 2)
        ax2.plot(x_norm, y_norm, 'r-', linewidth=2, label='Normal Distribution')
        ax2.legend()
        
        # Q-Q plot (approximate)
        ax3 = axes[0, 2]
        sorted_residuals = np.sort(results_df['residual'])
        n = len(sorted_residuals)
        theoretical_quantiles = np.random.normal(0, 1, n)
        theoretical_quantiles.sort()
        
        ax3.scatter(theoretical_quantiles, sorted_residuals, alpha=0.6)
        ax3.plot([theoretical_quantiles.min(), theoretical_quantiles.max()], 
                [sorted_residuals.min(), sorted_residuals.max()], 'r--', alpha=0.8)
        ax3.set_xlabel('Theoretical Quantiles')
        ax3.set_ylabel('Sample Quantiles')
        ax3.set_title('Q-Q Plot')
        ax3.grid(True, alpha=0.3)
        
        # Residuals by stock
        ax4 = axes[1, 0]
        stock_residuals = [results_df[results_df['stock_symbol'] == stock]['residual'].values 
                          for stock in results_df['stock_symbol'].unique()]
        ax4.boxplot(stock_residuals, labels=results_df['stock_symbol'].unique())
        ax4.set_xlabel('Stock Symbol')
        ax4.set_ylabel('Residuals')
        ax4.set_title('Residuals by Stock')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Absolute residuals vs predicted
        ax5 = axes[1, 1]
        abs_residuals = np.abs(results_df['residual'])
        ax5.scatter(results_df['predicted'], abs_residuals, alpha=0.6)
        ax5.set_xlabel('Predicted Values')
        ax5.set_ylabel('Absolute Residuals')
        ax5.set_title('Absolute Residuals vs Predicted')
        ax5.grid(True, alpha=0.3)
        
        # Residuals statistics
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Calculate residual statistics
        stats_text = f"""
        Residual Statistics:
        
        Mean: {results_df['residual'].mean():.6f}
        Std Dev: {results_df['residual'].std():.6f}
        Min: {results_df['residual'].min():.6f}
        Max: {results_df['residual'].max():.6f}
        
        Skewness: {results_df['residual'].skew():.4f}
        Kurtosis: {results_df['residual'].kurtosis():.4f}
        
        95% of residuals within:
        [{results_df['residual'].quantile(0.025):.6f}, 
         {results_df['residual'].quantile(0.975):.6f}]
        """
        
        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.plots_dir / "residuals_analysis.png"
        plt.savefig(save_path, dpi=self.config.visualization.get('dpi', 300), 
                    bbox_inches='tight')
        plt.close()
        
        print(f"✅ Residuals analysis plot saved to: {save_path}")
    
    def plot_correlation_matrix(self, correlation_matrix: pd.DataFrame) -> None:
        """Plot feature correlation matrix"""
        
        print("📊 Creating correlation matrix plot...")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 12))
        
        # Create heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', 
                   center=0, square=True, ax=ax, cbar_kws={"shrink": .8})
        
        ax.set_title('Feature Correlation Matrix')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        save_path = self.plots_dir / "correlation_matrix.png"
        plt.savefig(save_path, dpi=self.config.visualization.get('dpi', 300), 
                    bbox_inches='tight')
        plt.close()
        
        print(f"✅ Correlation matrix plot saved to: {save_path}")
    
    def create_model_comparison_plot(self, comparison_data: Dict[str, Dict[str, float]]) -> None:
        """Create model comparison visualization"""
        
        print("📊 Creating model comparison plot...")
        
        # Convert to DataFrame
        comparison_df = pd.DataFrame(comparison_data).T
        comparison_df.reset_index(inplace=True)
        comparison_df.rename(columns={'index': 'model'}, inplace=True)
        
        # Create subplot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        metrics = ['mae', 'mse', 'rmse', 'r2']
        titles = ['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error', 'R² Score']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i//2, i%2]
            if metric in comparison_df.columns:
                bars = ax.bar(comparison_df['model'], comparison_df[metric])
                ax.set_title(title)
                ax.set_ylabel(metric.upper())
                ax.tick_params(axis='x', rotation=45)
                
                # Color bars
                if metric == 'r2':
                    colors = plt.cm.viridis(comparison_df[metric] / comparison_df[metric].max())
                else:
                    colors = plt.cm.viridis_r(comparison_df[metric] / comparison_df[metric].max())
                
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.plots_dir / "model_comparison.png"
        plt.savefig(save_path, dpi=self.config.visualization.get('dpi', 300), 
                    bbox_inches='tight')
        plt.close()
        
        print(f"✅ Model comparison plot saved to: {save_path}")