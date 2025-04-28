import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional
import argparse

class ResultsAnalyzer:
    def __init__(self, metrics_dir: str, figures_dir: str):
        """
        Initialize the ResultsAnalyzer with paths to metrics and output directories.
        
        Args:
            metrics_dir (str): Directory containing training metrics JSON files
            figures_dir (str): Directory to save generated figures
        """
        self.metrics_dir = Path(metrics_dir)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Style configuration
        try:
            plt.style.use('seaborn')
        except:
            # Fallback to a basic style if seaborn is not available
            plt.style.use('default')
            # Set some basic styling to make plots look better
            plt.rcParams['figure.figsize'] = [12, 6]
            plt.rcParams['axes.grid'] = True
            plt.rcParams['grid.alpha'] = 0.3
        
        # Custom color scheme that works with both seaborn and default styles
        self.colors = {
            'baseline': '#1f77b4',  # Blue
            'augmented': '#2ca02c',  # Green
            'baseline_std': '#9ecae1',
            'augmented_std': '#a1d99b'
        }

    def load_metrics(self, prefix: str) -> Dict:
        """
        Load metrics from JSON files with given prefix.
        
        Args:
            prefix (str): File prefix ('baseline_' or 'augmented_')
            
        Returns:
            Dict containing loaded metrics
        """
        metrics = {}
        
        # Load training history
        history_path = self.metrics_dir / f"{prefix}training_history.json"
        if history_path.exists():
            with open(history_path) as f:
                metrics['history'] = json.load(f)
                
        # Load final metrics
        final_path = self.metrics_dir / f"{prefix}final_metrics.json"
        if final_path.exists():
            with open(final_path) as f:
                metrics['final'] = json.load(f)
                
        # Load CV summary if exists
        cv_path = self.metrics_dir / f"{prefix}cv_summary.json"
        if cv_path.exists():
            with open(cv_path) as f:
                metrics['cv'] = json.load(f)
                
        return metrics

    def plot_training_comparison(self, baseline_metrics: Dict, augmented_metrics: Dict):
        """Plot training metrics comparison between baseline and augmented models."""
        metrics_to_plot = [
            ('acc', 'Accuracy'),
            ('loss', 'Loss')
        ]
        
        for metric, title in metrics_to_plot:
            plt.figure(figsize=(12, 6))
            
            # Plot baseline
            if 'history' in baseline_metrics:
                plt.plot(baseline_metrics['history'][f'train_{metric}'], 
                        label=f'Baseline Train', color=self.colors['baseline'])
                plt.plot(baseline_metrics['history'][f'val_{metric}'], 
                        label=f'Baseline Val', color=self.colors['baseline'], linestyle='--')
            
            # Plot augmented
            if 'history' in augmented_metrics:
                plt.plot(augmented_metrics['history'][f'train_{metric}'], 
                        label=f'Augmented Train', color=self.colors['augmented'])
                plt.plot(augmented_metrics['history'][f'val_{metric}'], 
                        label=f'Augmented Val', color=self.colors['augmented'], linestyle='--')
            
            plt.title(f'Training {title} Comparison')
            plt.xlabel('Epoch')
            plt.ylabel(title)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            save_path = self.figures_dir / f'comparison_{metric}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

    def plot_cv_comparison(self, baseline_metrics: Dict, augmented_metrics: Dict):
        """Plot cross-validation results comparison."""
        if 'cv' not in baseline_metrics or 'cv' not in augmented_metrics:
            return
        
        metrics = ['accuracy', 'weighted_precision', 'weighted_recall', 'weighted_f1_score']
        
        # Prepare data for plotting
        data = []
        for model_type, metrics_dict in [('Baseline', baseline_metrics), ('Augmented', augmented_metrics)]:
            cv_results = metrics_dict['cv']
            for metric in metrics:
                mean = cv_results['average'][metric]
                std = cv_results['std_dev'][metric]
                data.append({
                    'Model': model_type,
                    'Metric': metric.replace('weighted_', '').replace('_', ' ').title(),
                    'Value': mean,
                    'Std': std
                })
        
        df = pd.DataFrame(data)
        
        # Create grouped bar plot
        plt.figure(figsize=(12, 6))
        bar_width = 0.35
        index = np.arange(len(metrics))
        
        baseline_mask = df['Model'] == 'Baseline'
        augmented_mask = df['Model'] == 'Augmented'
        
        plt.bar(index - bar_width/2, df[baseline_mask]['Value'], 
                bar_width, label='Baseline', color=self.colors['baseline'],
                yerr=df[baseline_mask]['Std'], capsize=5)
        plt.bar(index + bar_width/2, df[augmented_mask]['Value'], 
                bar_width, label='Augmented', color=self.colors['augmented'],
                yerr=df[augmented_mask]['Std'], capsize=5)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Cross-Validation Results Comparison')
        plt.xticks(index, df[baseline_mask]['Metric'])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = self.figures_dir / 'cv_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def generate_summary_report(self, baseline_metrics: Dict, augmented_metrics: Dict) -> str:
        """Generate a text summary comparing baseline and augmented model performance."""
        report = ["=== Model Performance Comparison Report ===\n"]
        
        # Final metrics comparison
        if 'final' in baseline_metrics and 'final' in augmented_metrics:
            report.append("Final Test Set Performance:")
            baseline_final = baseline_metrics['final']['metrics']
            augmented_final = augmented_metrics['final']['metrics']
            
            metrics_to_report = [
                ('accuracy', 'Accuracy'),
                ('weighted_precision', 'Precision'),
                ('weighted_recall', 'Recall'),
                ('weighted_f1_score', 'F1 Score')
            ]
            
            for metric_key, metric_name in metrics_to_report:
                baseline_value = baseline_final[metric_key]
                augmented_value = augmented_final[metric_key]
                improvement = (augmented_value - baseline_value) / baseline_value * 100
                
                report.append(f"\n{metric_name}:")
                report.append(f"  Baseline:  {baseline_value:.4f}")
                report.append(f"  Augmented: {augmented_value:.4f}")
                report.append(f"  Change:    {improvement:+.1f}%")
        
        # Cross-validation summary
        if 'cv' in baseline_metrics and 'cv' in augmented_metrics:
            report.append("\nCross-Validation Results:")
            baseline_cv = baseline_metrics['cv']['average']
            augmented_cv = augmented_metrics['cv']['average']
            
            for metric in ['accuracy', 'weighted_precision', 'weighted_recall', 'weighted_f1_score']:
                name = metric.replace('weighted_', '').replace('_', ' ').title()
                baseline_value = baseline_cv[metric]
                baseline_std = baseline_metrics['cv']['std_dev'][metric]
                augmented_value = augmented_cv[metric]
                augmented_std = augmented_metrics['cv']['std_dev'][metric]
                
                report.append(f"\n{name}:")
                report.append(f"  Baseline:  {baseline_value:.4f} ± {baseline_std:.4f}")
                report.append(f"  Augmented: {augmented_value:.4f} ± {augmented_std:.4f}")
        
        report = '\n'.join(report)
        
        # Save report to file
        report_path = self.figures_dir / 'comparison_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        
        return report

def main():
    parser = argparse.ArgumentParser(description='Analyze and compare baseline and augmented model results')
    parser.add_argument('--metrics-dir', type=str, default='./results/metrics',
                        help='Directory containing metrics files')
    parser.add_argument('--figures-dir', type=str, default='./results/analysis',
                        help='Directory to save analysis outputs')
    args = parser.parse_args()
    
    analyzer = ResultsAnalyzer(args.metrics_dir, args.figures_dir)
    
    # Load metrics
    baseline_metrics = analyzer.load_metrics('baseline_')
    augmented_metrics = analyzer.load_metrics('augmented_')
    
    if not baseline_metrics or not augmented_metrics:
        print("Error: Could not find required metrics files.")
        return
    
    # Generate visualizations
    print("Generating training comparison plots...")
    analyzer.plot_training_comparison(baseline_metrics, augmented_metrics)
    
    print("Generating cross-validation comparison plots...")
    analyzer.plot_cv_comparison(baseline_metrics, augmented_metrics)
    
    # Generate and print summary report
    print("\nGenerating summary report...")
    report = analyzer.generate_summary_report(baseline_metrics, augmented_metrics)
    print("\n" + report)
    
    print(f"\nAnalysis complete. Results saved to {args.figures_dir}")

if __name__ == '__main__':
    main() 