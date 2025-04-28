import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional
import argparse
import random
from PIL import Image
import cv2 # For resizing and grayscale conversion in SSIM/GradCAM
from tqdm import tqdm  # Add tqdm import for progress bars

# SSIM calculation
from skimage.metrics import structural_similarity as ssim

# Grad-CAM
import torch
import torch.nn
import torchvision.transforms as T
from torchvision.models import resnet50 # Assuming ResNet50 is used
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

# Project imports (assuming they are accessible)
# Need to adjust path if analyze_results.py is run from root or src
try:
    # If run from root
    from src.classifier import create_resnet50_baseline
    from src.data_loader import RSNAPneumoniaDataset, SyntheticDataset, data_transforms
    from src.utils import check_create_dir
except ImportError:
    # If run from src directory
    print("Running from src directory, adjusting imports...")
    # No need to modify sys.path if using relative imports correctly
    # import sys
    # sys.path.append('..') # Add project root to path
    from classifier import create_resnet50_baseline # Use relative import
    from data_loader import RSNAPneumoniaDataset, SyntheticDataset, data_transforms # Use relative import
    from utils import check_create_dir # Use relative import

# Inverse transform for visualization
INV_NORMALIZE = T.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

def deprocess_image(tensor):
    """Convert a tensor image back to a NumPy array for visualization."""
    tensor = INV_NORMALIZE(tensor) # Apply inverse normalization
    img_np = tensor.squeeze().cpu().numpy() # Remove batch dim, move to CPU, convert to NumPy
    img_np = np.transpose(img_np, (1, 2, 0)) # Change from C, H, W to H, W, C
    img_np = np.clip(img_np, 0, 1) # Clip values to [0, 1]
    img_np = (img_np * 255).astype(np.uint8) # Scale to [0, 255] and convert to uint8
    return img_np

class ResultsAnalyzer:
    def __init__(self, metrics_dir: str, analysis_dir: str, model_dir: str,
                 data_dir: str, synthetic_dir: str, device: torch.device):
        """
        Initialize the ResultsAnalyzer.
        
        Args:
            metrics_dir (str): Directory containing metrics JSON files.
            analysis_dir (str): Directory to save analysis outputs (plots, reports).
            model_dir (str): Directory containing saved model checkpoints.
            data_dir (str): Path to the processed (real) dataset directory.
            synthetic_dir (str): Path to the synthetic image directory.
            device: PyTorch device ('cuda' or 'cpu').
        """
        self.metrics_dir = Path(metrics_dir)
        self.analysis_dir = Path(analysis_dir)
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.synthetic_dir = Path(synthetic_dir)
        self.device = device
        
        # Define figures_dir before using it
        self.figures_dir = self.analysis_dir # Save all figures in the analysis dir
        
        check_create_dir(self.analysis_dir)
        check_create_dir(self.figures_dir)

        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # Style configuration
        try:
            plt.style.use('seaborn-v0_8-darkgrid') # Use a specific seaborn style
        except:
            plt.style.use('default')
            plt.rcParams['figure.figsize'] = [12, 6]
            plt.rcParams['axes.grid'] = True
            plt.rcParams['grid.alpha'] = 0.3
        
        # Custom color scheme that works with both seaborn and default styles
        self.colors = {
            'baseline': '#1f77b4',  # Blue
            'augmented': '#2ca02c',  # Green
            'curriculum': '#ff7f0e', # Add color for curriculum
            'baseline_std': '#9ecae1',
            'augmented_std': '#a1d99b',
            'curriculum_std': '#ffbb78'
        }

    def load_metrics(self, prefix: str) -> Optional[Dict]:
        """
        Load metrics from JSON files with given prefix.
        
        Args:
            prefix (str): File prefix ('baseline_', 'augmented_', 'curriculum_')
            
        Returns:
            Dict containing loaded metrics or None if essential files are missing
        """
        metrics = {}
        essential_missing = False

        # Load training history
        history_path = self.metrics_dir / f"{prefix}training_history.json"
        if history_path.exists():
            with open(history_path) as f:
                metrics['history'] = json.load(f)
        else:
            print(f"Warning: Training history not found: {history_path}")
            # Decide if this is critical. Let's assume it is for plotting.
            # essential_missing = True

        # Load final metrics (optional for some analyses)
        final_path = self.metrics_dir / f"{prefix}final_metrics.json"
        if final_path.exists():
            with open(final_path) as f:
                metrics['final'] = json.load(f)
        else:
            print(f"Info: Final metrics not found: {final_path}")

        # Load CV summary if exists
        cv_path = self.metrics_dir / f"{prefix}cv_summary.json"
        if cv_path.exists():
            with open(cv_path) as f:
                metrics['cv'] = json.load(f)
        else:
            print(f"Info: CV summary not found: {cv_path}")
            # If CV summary is missing, but history exists, we might still plot single run
            if 'history' not in metrics:
                essential_missing = True # Need at least history or CV summary

        if essential_missing:
            print(f"Error: Essential metrics files missing for prefix '{prefix}'. Cannot proceed with analysis for this run.")
            return None
        if not metrics:
            print(f"Warning: No metrics files found for prefix '{prefix}'.")
            return None

        return metrics

    def plot_training_comparison(self, metrics_dict: Dict[str, Dict]):
        """Plot training metrics comparison for multiple model types (baseline, augmented, etc.)."""
        metrics_to_plot = [
            ('acc', 'Accuracy'),
            ('loss', 'Loss'),
            ('synthetic_ratio', 'Synthetic Ratio') # Add ratio plot
        ]

        valid_runs = {k: v for k, v in metrics_dict.items() if v and 'history' in v}
        if not valid_runs:
            print("No valid training history found to plot comparisons.")
            return

        for metric, title in metrics_to_plot:
            plt.figure(figsize=(12, 6))
            has_data = False

            for run_name, run_metrics in valid_runs.items():
                history = run_metrics['history']
                color = self.colors.get(run_name, '#808080') # Default gray
                linestyle = '--' if 'val' in metric else '-'
                label_prefix = run_name.replace("_", " ").title()

                if metric == 'synthetic_ratio':
                    if 'synthetic_ratio' in history and any(history['synthetic_ratio']): # Only plot if non-zero ratios exist
                        ratio_key = 'synthetic_ratio'
                        if ratio_key in history and len(history[ratio_key]) > 0:
                            epochs = range(1, len(history[ratio_key]) + 1)
                            plt.plot(epochs, history[ratio_key], label=f'{label_prefix} Ratio', color=color, linestyle='-.' if linestyle == '-' else '-')
                            has_data = True
                else:
                    train_key = f'train_{metric}'
                    val_key = f'val_{metric}'
                    if train_key in history and val_key in history:
                        epochs = range(1, len(history[train_key]) + 1)
                        plt.plot(epochs, history[train_key], label=f'{label_prefix} Train', color=color, linestyle='-')
                        plt.plot(epochs, history[val_key], label=f'{label_prefix} Val', color=color, linestyle='--')
                        has_data = True

            if not has_data:
                plt.close()
                print(f"No data found for metric '{title}' comparison.")
                continue

            plt.title(f'Training {title} Comparison')
            plt.xlabel('Epoch')
            plt.ylabel(title)
            plt.legend()
            plt.grid(True, alpha=0.3)

            save_path = self.figures_dir / f'comparison_{metric}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {title} comparison plot to {save_path}")

    def plot_cv_comparison(self, metrics_dict: Dict[str, Dict]):
        """Plot cross-validation results comparison."""
        valid_runs = {k: v for k, v in metrics_dict.items() if v and 'cv' in v}
        if len(valid_runs) < 1:
            print("No valid cross-validation results found to plot comparison.")
            return

        metrics = ['accuracy', 'weighted_precision', 'weighted_recall', 'weighted_f1_score']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        num_metrics = len(metrics)
        num_runs = len(valid_runs)

        # Prepare data for plotting
        data = []
        for run_name, run_metrics in valid_runs.items():
            cv_results = run_metrics['cv']
            if 'average' not in cv_results or 'std_dev' not in cv_results:
                print(f"Warning: Missing 'average' or 'std_dev' in CV results for {run_name}")
                continue

            for metric_key, metric_label in zip(metrics, metric_labels):
                mean = cv_results['average'].get(metric_key, np.nan)
                std = cv_results['std_dev'].get(metric_key, np.nan)
                data.append({
                    'Model': run_name.replace("_", " ").title(),
                    'Metric': metric_label,
                    'Value': mean,
                    'Std': std
                })

        if not data:
            print("No data prepared for CV comparison plot.")
            return

        df = pd.DataFrame(data)
        df = df.dropna(subset=['Value']) # Remove metrics that were entirely missing

        if df.empty:
            print("DataFrame empty after dropping NaNs for CV comparison plot.")
            return

        # Create grouped bar plot
        plt.figure(figsize=(max(10, num_metrics * num_runs * 0.8), 6))
        bar_width = 0.8 / num_runs
        metric_names = df['Metric'].unique()
        index = np.arange(len(metric_names))

        for i, run_name in enumerate(df['Model'].unique()):
            run_df = df[df['Model'] == run_name]
            # Align run_df with metric_names order
            run_df = run_df.set_index('Metric').reindex(metric_names).reset_index()
            color = self.colors.get(run_name.lower().replace(" ", "_"), f'C{i}')
            std_color = self.colors.get(f"{run_name.lower().replace(' ', '_')}_std", f'C{i}')

            plt.bar(index - (num_runs/2 - 0.5 - i) * bar_width, run_df['Value'],
                    bar_width, label=run_name, color=color,
                    yerr=run_df['Std'], capsize=5, alpha=0.8)

        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Cross-Validation Results Comparison (Mean ± Std Dev)')
        plt.xticks(index, metric_names)
        plt.ylim(bottom=max(0, df['Value'].min() - df['Std'].max() - 0.1)) # Adjust bottom limit
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        save_path = self.figures_dir / 'cv_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved CV comparison plot to {save_path}")

    def generate_summary_report(self, metrics_dict: Dict[str, Dict]) -> str:
        """Generate a text summary comparing model performance for available runs."""
        report_lines = ["=== Model Performance Comparison Report ===\n"]
        valid_runs = {k: v for k, v in metrics_dict.items() if v}

        metrics_to_report = [
            ('accuracy', 'Accuracy'),
            ('weighted_precision', 'Precision (Weighted)'),
            ('weighted_recall', 'Recall (Weighted)'),
            ('weighted_f1_score', 'F1 Score (Weighted)')
        ]

        # --- Final Test Set Performance --- #
        final_perf_available = {name: run['final']['metrics'] for name, run in valid_runs.items() if 'final' in run and 'metrics' in run['final']}
        if final_perf_available:
            report_lines.append("\n--- Final Test Set Performance ---")
            baseline_final = final_perf_available.get('baseline', None)

            for name, metrics in final_perf_available.items():
                report_lines.append(f"\n* {name.replace('_', ' ').title()}:")
                for key, label in metrics_to_report:
                    value = metrics.get(key, 'N/A')
                    line = f"  - {label:<20}: {value:.4f}" if isinstance(value, float) else f"  - {label:<20}: {value}"
                    # Compare to baseline if baseline exists and this isn't baseline
                    if baseline_final and name != 'baseline' and key in baseline_final:
                        baseline_value = baseline_final[key]
                        if isinstance(value, float) and isinstance(baseline_value, float) and baseline_value != 0:
                            improvement = (value - baseline_value) / baseline_value * 100
                            line += f" ({improvement:+.1f}% vs Baseline)"
                        elif isinstance(value, float) and isinstance(baseline_value, float):
                            line += " (Baseline: 0)"
                    report_lines.append(line)
            report_lines.append("") # Add spacing
        else:
            report_lines.append("\n--- Final Test Set Performance: No data found ---")

        # --- Cross-Validation Performance --- #
        cv_perf_available = {name: run['cv'] for name, run in valid_runs.items() if 'cv' in run and 'average' in run['cv'] and 'std_dev' in run['cv']}
        if cv_perf_available:
            report_lines.append("\n--- Cross-Validation Performance (Average ± Std Dev) ---")
            baseline_cv_avg = cv_perf_available.get('baseline', {}).get('average', None)

            for name, cv_data in cv_perf_available.items():
                report_lines.append(f"\n* {name.replace('_', ' ').title()}:")
                avg_metrics = cv_data['average']
                std_metrics = cv_data['std_dev']
                for key, label in metrics_to_report:
                    avg_value = avg_metrics.get(key, 'N/A')
                    std_value = std_metrics.get(key, 'N/A')
                    line = f"  - {label:<20}: {avg_value:.4f} ± {std_value:.4f}" if isinstance(avg_value, float) and isinstance(std_value, float) else f"  - {label:<20}: {avg_value} ± {std_value}"
                    # Compare average to baseline average if available
                    if baseline_cv_avg and name != 'baseline' and key in baseline_cv_avg:
                        baseline_avg = baseline_cv_avg[key]
                        if isinstance(avg_value, float) and isinstance(baseline_avg, float) and baseline_avg != 0:
                            improvement = (avg_value - baseline_avg) / baseline_avg * 100
                            line += f" ({improvement:+.1f}% vs Baseline Avg)"
                        elif isinstance(avg_value, float) and isinstance(baseline_avg, float):
                            line += " (Baseline Avg: 0)"
                    report_lines.append(line)
            report_lines.append("") # Add spacing
        else:
            report_lines.append("\n--- Cross-Validation Performance: No data found ---")

        report = '\n'.join(report_lines)

        # Save report to file
        report_path = self.analysis_dir / 'comparison_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Saved comparison report to {report_path}")

        return report

    # --- SSIM Analysis --- #
    def calculate_ssim_distribution(self, num_real_samples=100, num_synthetic_samples=500):
        """Calculates SSIM between synthetic images and a sample of real positive images."""
        print("\n--- Calculating SSIM Distribution --- ")
        try:
            # Define transform for SSIM (resize, grayscale, numpy)
            ssim_transform = T.Compose([
                T.Resize((224, 224)),
                T.Grayscale(),
                T.ToTensor() # ToTensor scales to [0, 1]
            ])

            # Load real dataset (only need metadata and paths)
            real_metadata_path = self.data_dir / 'stage2_train_metadata.csv'
            if not real_metadata_path.exists():
                print(f"Error: Real metadata not found at {real_metadata_path}")
                return
            real_df = pd.read_csv(real_metadata_path)
            real_df['label'] = (real_df['class'] == 'Lung Opacity').astype(int)
            positive_real_df = real_df[real_df['label'] == 1]

            if positive_real_df.empty:
                print("Error: No positive real images found in metadata.")
                return

            # Sample real positive images
            num_real_to_sample = min(num_real_samples, len(positive_real_df))
            real_samples_df = positive_real_df.sample(n=num_real_to_sample, random_state=42)
            real_images_np = []
            print(f"Loading {num_real_to_sample} real positive images for SSIM reference...")
            for patient_id in tqdm(real_samples_df['patientId'], desc="Loading Real Images"):
                img_path = self.data_dir / 'Training' / 'Images' / f"{patient_id}.png"
                try:
                    img = Image.open(img_path).convert('RGB') # Open as RGB first
                    img_tensor = ssim_transform(img)
                    img_np = img_tensor.squeeze().numpy() # Remove channel dim for grayscale
                    real_images_np.append(img_np)
                except Exception as e:
                    print(f"Warning: Could not load real image {img_path}: {e}")
            if not real_images_np:
                print("Error: Failed to load any real reference images.")
                return

            # Load synthetic images
            synthetic_files = list(self.synthetic_dir.glob('*.png'))
            if not synthetic_files:
                print(f"Error: No synthetic images found in {self.synthetic_dir}")
                return
            num_synthetic_to_sample = min(num_synthetic_samples, len(synthetic_files))
            synthetic_files_sampled = random.sample(synthetic_files, num_synthetic_to_sample)

            avg_ssim_scores = []
            print(f"Calculating average SSIM for {num_synthetic_to_sample} synthetic images...")
            for synth_path in tqdm(synthetic_files_sampled, desc="Calculating SSIM"):
                try:
                    synth_img = Image.open(synth_path).convert('RGB')
                    synth_tensor = ssim_transform(synth_img)
                    synth_np = synth_tensor.squeeze().numpy()

                    current_ssim_scores = []
                    for real_np in real_images_np:
                        # Ensure data_range is appropriate (images are in [0, 1])
                        score = ssim(synth_np, real_np, data_range=1.0)
                        current_ssim_scores.append(score)

                    if current_ssim_scores:
                        avg_ssim_scores.append(np.mean(current_ssim_scores))
                except Exception as e:
                    print(f"Warning: Could not process synthetic image {synth_path}: {e}")

            if not avg_ssim_scores:
                print("Error: Failed to calculate any SSIM scores.")
                return

            # Plot histogram
            plt.figure(figsize=(10, 6))
            sns.histplot(avg_ssim_scores, kde=True, bins=30)
            mean_ssim = np.mean(avg_ssim_scores)
            median_ssim = np.median(avg_ssim_scores)
            plt.title(f'Distribution of Average SSIM (Synthetic vs. {num_real_to_sample} Real Positives)\nMean: {mean_ssim:.3f}, Median: {median_ssim:.3f}')
            plt.xlabel('Average SSIM Score')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            ssim_plot_path = self.analysis_dir / 'ssim_distribution.png'
            plt.savefig(ssim_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved SSIM distribution plot to {ssim_plot_path}")

        except Exception as e:
            print(f"An error occurred during SSIM calculation: {e}")
            import traceback
            traceback.print_exc()

    # --- Grad-CAM Analysis --- #
    def generate_grad_cam_comparison(self, num_samples=3):
        """Generates Grad-CAM visualizations comparing baseline and augmented models."""
        print("\n--- Generating Grad-CAM Comparison --- ")

        try:
            # --- Load Models --- #
            models = {}
            target_layers = {}
            for prefix in ['baseline_', 'augmented_', 'curriculum_']: # Add curriculum if exists
                model_path = self.model_dir / f"{prefix}resnet50.pth"
                run_name = prefix[:-1]
                if model_path.exists():
                    print(f"Loading model: {model_path}")
                    model = create_resnet50_baseline(num_classes=2)
                    try:
                        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
                        model = model.to(self.device)  # Move model to correct device
                        model.eval()  # Set to eval mode
                        models[run_name] = model
                        # Get the last convolutional layer for ResNet50
                        target_layers[run_name] = [model.layer4[-1].conv3]  # Use the last conv layer in the last bottleneck
                    except Exception as e:
                        print(f"Warning: Failed to load model {model_path}: {e}")
                else:
                    print(f"Info: Model file not found, skipping Grad-CAM for {run_name}: {model_path}")

            if len(models) < 1:
                print("Error: No models loaded successfully for Grad-CAM.")
                return

            # --- Prepare CAM Objects --- #
            cams = {}
            for name, model in models.items():
                try:
                    cams[name] = GradCAM(model=model, 
                                       target_layers=target_layers[name])  # Removed use_cuda parameter
                except Exception as e:
                    print(f"Warning: Failed to create GradCAM for {name}: {e}")

            # --- Load Sample Images --- #
            real_metadata_path = self.data_dir / 'stage2_train_metadata.csv'
            if not real_metadata_path.exists(): return # Already checked in SSIM
            real_df = pd.read_csv(real_metadata_path)
            real_df['label'] = (real_df['class'] == 'Lung Opacity').astype(int)

            positive_samples = real_df[real_df['label'] == 1].sample(n=num_samples, random_state=43).to_dict('records')
            negative_samples = real_df[real_df['label'] == 0].sample(n=num_samples, random_state=44).to_dict('records')

            synthetic_files = list(self.synthetic_dir.glob('*.png'))
            synthetic_samples = []
            if synthetic_files:
                sampled_synth_files = random.sample(synthetic_files, min(num_samples, len(synthetic_files)))
                synthetic_samples = [{'patientId': f.stem, 'path': f, 'label': 1, 'type': 'synthetic'} for f in sampled_synth_files]
            else:
                print("Warning: No synthetic images found for Grad-CAM.")

            sample_list = (
                [{**s, 'type': 'real_positive'} for s in positive_samples] +
                [{**s, 'type': 'real_negative'} for s in negative_samples] +
                synthetic_samples
            )

            # Preprocessing transform (use the validation/test transform)
            preprocess = data_transforms['test']

            # --- Generate CAMs for Samples --- #
            print(f"Generating Grad-CAM for {len(sample_list)} samples...")
            for sample in tqdm(sample_list, desc="Generating CAMs"):
                patient_id = sample['patientId']
                label = sample['label']
                sample_type = sample['type']

                if sample_type == 'synthetic':
                    img_path = sample['path']
                else:
                    img_path = self.data_dir / 'Training' / 'Images' / f"{patient_id}.png"

                try:
                    # Ensure image is loaded and converted to RGB properly
                    rgb_img = Image.open(img_path).convert('RGB')
                    rgb_img_resized = rgb_img.resize((224, 224))
                    rgb_img_np = np.array(rgb_img_resized) / 255.0

                    # Ensure input tensor is properly preprocessed and on correct device
                    input_tensor = preprocess(rgb_img).unsqueeze(0)
                    input_tensor = input_tensor.to(self.device)
                    
                    # Create figure before generating CAMs
                    fig, axes = plt.subplots(1, 1 + len(models), figsize=(5 * (1 + len(models)), 5))
                    if len(models) == 1:
                        axes = [axes]
                    elif not isinstance(axes, np.ndarray):
                        axes = [axes]

                    # Plot original image
                    axes[0].imshow(rgb_img_np)
                    axes[0].set_title(f"Original ({sample_type})\nID: {patient_id}, Label: {label}")
                    axes[0].axis('off')

                    # Generate CAMs for each model
                    for i, (run_name, model) in enumerate(models.items()):
                        # Get model prediction
                        with torch.no_grad():  # No need for gradients during prediction
                            output = model(input_tensor)
                            pred_label = output.argmax(dim=1).item()
                        
                        # Enable gradients only for GradCAM
                        input_tensor.requires_grad = True
                        
                        # Use predicted label for visualization
                        target_category = [ClassifierOutputTarget(pred_label)]
                        
                        # Generate CAM
                        try:
                            grayscale_cam = cams[run_name](input_tensor=input_tensor,
                                                         targets=target_category,
                                                         eigen_smooth=True)  # Enable eigen smoothing
                            grayscale_cam = grayscale_cam[0, :]
                            
                            # Create visualization
                            visualization = show_cam_on_image(rgb_img_np, grayscale_cam, use_rgb=True)
                            
                            # Plot CAM
                            axes[i+1].imshow(visualization)
                            axes[i+1].set_title(f"{run_name.title()} CAM\nPred: {pred_label}, True: {label}")
                            axes[i+1].axis('off')
                        except Exception as e:
                            print(f"Warning: Failed to generate CAM for {run_name}: {e}")
                            axes[i+1].imshow(rgb_img_np)  # Show original image if CAM fails
                            axes[i+1].set_title(f"{run_name.title()} CAM Failed\nPred: {pred_label}, True: {label}")
                            axes[i+1].axis('off')

                    plt.tight_layout()
                    cam_save_path = self.analysis_dir / f"gradcam_{sample_type}_{patient_id}.png"
                    plt.savefig(cam_save_path, dpi=150, bbox_inches='tight')
                    plt.close()

                except Exception as e:
                    print(f"Warning: Failed Grad-CAM for {patient_id} ({sample_type}): {e}")
                    import traceback
                    traceback.print_exc()

            print(f"Finished Grad-CAM generation. Images saved in {self.analysis_dir}")

        except ImportError:
            print("Error: pytorch-grad-cam or dependencies not found. Skipping Grad-CAM analysis.")
            print("Install with: pip install grad-cam")
        except Exception as e:
            print(f"An error occurred during Grad-CAM generation: {e}")
            import traceback
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Analyze and compare baseline and augmented model results')
    parser.add_argument('--metrics-dir', type=str, default='./results/metrics',
                        help='Directory containing metrics files (default: ./results/metrics)')
    parser.add_argument('--analysis-dir', type=str, default='./results/analysis',
                        help='Directory to save analysis outputs (default: ./results/analysis)')
    parser.add_argument('--model-dir', type=str, default='./models',
                        help='Directory containing saved model checkpoints (default: ./models)')
    parser.add_argument('--data-dir', type=str, default='./data/processed',
                        help='Path to the processed (real) dataset directory (default: ./data/processed)')
    parser.add_argument('--synthetic-dir', type=str, default='./data/synthetic',
                        help='Path to the synthetic images directory (default: ./data/synthetic)')
    parser.add_argument('--num-ssim-real', type=int, default=100,
                        help='Number of real positive samples for SSIM comparison (default: 100)')
    parser.add_argument('--num-ssim-synth', type=int, default=500,
                         help='Number of synthetic samples for SSIM calculation (default: 500)')
    parser.add_argument('--num-gradcam-samples', type=int, default=3,
                        help='Number of samples per category (real pos/neg, synth) for Grad-CAM (default: 3)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    parser.add_argument('--skip-plots', action='store_true', help='Skip generating training/CV plots')
    parser.add_argument('--skip-ssim', action='store_true', help='Skip SSIM calculation')
    parser.add_argument('--skip-gradcam', action='store_true', help='Skip Grad-CAM generation')

    args = parser.parse_args()

    # Setup device
    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
    print(f"Using device: {device}")

    analyzer = ResultsAnalyzer(args.metrics_dir, args.analysis_dir, args.model_dir,
                               args.data_dir, args.synthetic_dir, device)

    # Load metrics for all potential runs
    metrics_data = {}
    for prefix in ['baseline_', 'augmented_', 'curriculum_']:
        run_name = prefix[:-1]
        loaded = analyzer.load_metrics(prefix)
        if loaded:
            metrics_data[run_name] = loaded

    if not metrics_data:
        print("Error: No valid metrics loaded. Aborting analysis.")
        return

    # --- Generate Comparisons --- #
    if not args.skip_plots:
        print("\nGenerating training comparison plots...")
        analyzer.plot_training_comparison(metrics_data)

        print("\nGenerating cross-validation comparison plots...")
        analyzer.plot_cv_comparison(metrics_data)

    # --- Generate Summary Report --- #
    print("\nGenerating summary report...")
    report = analyzer.generate_summary_report(metrics_data)
    print("\n" + report)

    # --- Generate SSIM Plot --- #
    if not args.skip_ssim:
        analyzer.calculate_ssim_distribution(args.num_ssim_real, args.num_ssim_synth)

    # --- Generate Grad-CAM Plots --- #
    if not args.skip_gradcam:
        analyzer.generate_grad_cam_comparison(args.num_gradcam_samples)

    print(f"\nAnalysis complete. Outputs saved to {args.analysis_dir}")

if __name__ == '__main__':
    main() 