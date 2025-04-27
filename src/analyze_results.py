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
from skimage.metrics import structural_similarity as ssim
import torch
from torchvision import transforms
import sys
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from scipy import linalg
from tqdm import tqdm

# Get the absolute path of the project root directory
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR))  # Add project root to Python path

from src.data_loader import RSNAPneumoniaDataset  # Fixed class name

class InceptionV3Features(nn.Module):
    """Modified InceptionV3 for feature extraction."""
    def __init__(self):
        super(InceptionV3Features, self).__init__()
        inception = models.inception_v3(pretrained=True)
        self.block1 = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3, nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.block2 = nn.Sequential(
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.block3 = nn.Sequential(
            inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d,
            inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e
        )
        self.block4 = nn.Sequential(
            inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.view(x.size(0), -1)

class ResultsAnalyzer:
    def __init__(self, metrics_dir: str, analysis_dir: str, data_dir: Optional[str] = None, synthetic_dir: Optional[str] = None):
        """
        Initialize the ResultsAnalyzer with paths to metrics and output directories.
        
        Args:
            metrics_dir (str): Directory containing training metrics JSON files
            analysis_dir (str): Directory to save generated figures and reports
            data_dir (Optional[str]): Path to the processed real dataset directory (for SSIM)
            synthetic_dir (Optional[str]): Path to the synthetic image directory (for SSIM)
        """
        self.metrics_dir = Path(metrics_dir)
        self.analysis_dir = Path(analysis_dir)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = Path(data_dir) if data_dir else None
        self.synthetic_dir = Path(synthetic_dir) if synthetic_dir else None
        
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
        
        # Standard ImageNet transforms (assuming images are saved in 0-1 range or need normalization)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1), # SSIM often computed on grayscale
            transforms.ToTensor(),
            # Add normalization if images aren't already normalized
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])

        # Add inception model for FID calculation
        self.inception = InceptionV3Features().eval()
        if torch.cuda.is_available():
            self.inception = self.inception.cuda()
        
        # Update transform for FID calculation
        self.fid_transform = transforms.Compose([
            transforms.Resize(299),  # InceptionV3 input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

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
            
            save_path = self.analysis_dir / f'comparison_{metric}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

    def plot_cv_comparison(self, baseline_metrics: Dict, augmented_metrics: Dict):
        """Plot cross-validation results comparison."""
        if 'cv' not in baseline_metrics or 'cv' not in augmented_metrics:
            print("Skipping CV comparison plot: CV results not found for both models.")
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
        
        save_path = self.analysis_dir / 'cv_comparison.png'
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
        report_path = self.analysis_dir / 'comparison_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        
        return report

    def _load_image_paths(self, directory: Path, num_images: int, extensions=('.png', '.jpg', '.jpeg')) -> List[Path]:
        """Helper to load a random sample of image paths from a directory."""
        all_paths = [p for ext in extensions for p in directory.rglob(f'*{ext}')]
        if not all_paths:
            return []
        return random.sample(all_paths, min(num_images, len(all_paths)))

    def _calculate_ssim(self, img1_path: Path, img2_path: Path) -> Optional[float]:
        """Calculate SSIM between two images after applying transforms."""
        try:
            img1 = Image.open(img1_path).convert("RGB") # Ensure 3 channels for transform
            img2 = Image.open(img2_path).convert("RGB") # Ensure 3 channels for transform
            
            # Apply transforms (ensure grayscale if needed)
            img1_t = self.transform(img1).squeeze(0).numpy() # Remove channel dim for grayscale ssim
            img2_t = self.transform(img2).squeeze(0).numpy()
            
            # Ensure images have the same shape
            if img1_t.shape != img2_t.shape:
                 print(f"Warning: Skipping SSIM due to shape mismatch: {img1_path.name} {img1_t.shape} vs {img2_path.name} {img2_t.shape}")
                 return None
            
            # Calculate SSIM
            # data_range is max_val - min_val of the image dtype. For float images in [0, 1], it's 1.0
            score = ssim(img1_t, img2_t, data_range=1.0)
            return score
        except Exception as e:
            print(f"Error calculating SSIM between {img1_path.name} and {img2_path.name}: {e}")
            return None

    def calculate_activation_statistics(self, image_paths: List[Path], batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate mean and covariance of InceptionV3 activations."""
        all_features = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(image_paths), batch_size), desc="Calculating activation statistics"):
                batch_paths = image_paths[i:i + batch_size]
                batch_images = []
                
                for path in batch_paths:
                    img = Image.open(path).convert('RGB')
                    img = self.fid_transform(img)
                    batch_images.append(img)
                
                batch_tensor = torch.stack(batch_images)
                if torch.cuda.is_available():
                    batch_tensor = batch_tensor.cuda()
                
                features = self.inception(batch_tensor)
                all_features.append(features.cpu().numpy())
        
        all_features = np.concatenate(all_features, axis=0)
        mu = np.mean(all_features, axis=0)
        sigma = np.cov(all_features, rowvar=False)
        
        return mu, sigma

    def calculate_fid(self, real_paths: List[Path], synthetic_paths: List[Path], batch_size: int = 32) -> float:
        """Calculate Fréchet Inception Distance between real and synthetic images."""
        print("\nCalculating FID score...")
        
        # Calculate statistics for real and synthetic images
        mu_real, sigma_real = self.calculate_activation_statistics(real_paths, batch_size)
        mu_synthetic, sigma_synthetic = self.calculate_activation_statistics(synthetic_paths, batch_size)
        
        # Calculate FID
        diff = mu_real - mu_synthetic
        covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_synthetic), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid_score = diff.dot(diff) + np.trace(sigma_real + sigma_synthetic - 2 * covmean)
        return float(fid_score)

    def calculate_precision_recall(self, real_features: np.ndarray, synthetic_features: np.ndarray, k: int = 3) -> Tuple[float, float]:
        """Calculate precision and recall metrics using k-nearest neighbors."""
        print("\nCalculating Precision/Recall metrics...")
        
        def knn_precision_recall(real_dots: np.ndarray, synthetic_dots: np.ndarray) -> Tuple[float, float]:
            """Helper function for precision/recall calculation."""
            real_nearest = []
            synthetic_nearest = []
            
            for synthetic_dot in synthetic_dots:
                dist = np.sum((real_dots - synthetic_dot) ** 2, axis=1)
                real_nearest.append(np.partition(dist, k)[:k])
            
            for real_dot in real_dots:
                dist = np.sum((synthetic_dots - real_dot) ** 2, axis=1)
                synthetic_nearest.append(np.partition(dist, k)[:k])
            
            real_nearest = np.array(real_nearest)
            synthetic_nearest = np.array(synthetic_nearest)
            
            precision = (real_nearest < np.max(synthetic_nearest)).sum() / (real_nearest.size)
            recall = (synthetic_nearest < np.max(real_nearest)).sum() / (synthetic_nearest.size)
            
            return precision, recall
        
        return knn_precision_recall(real_features, synthetic_features)

    def analyze_synthetic_quality(self, num_compare: int = 100):
        """Enhanced analysis including SSIM, FID, and Precision/Recall metrics."""
        if not self.data_dir or not self.synthetic_dir:
            print("Skipping analysis: Real data or synthetic data directory not provided.")
            return
        if not self.data_dir.exists() or not self.synthetic_dir.exists():
            print("Skipping analysis: Real data or synthetic data directory does not exist.")
            return

        print(f"\nStarting comprehensive synthetic image quality analysis...")

        # Load real positive images (assuming PneumoniaDataset structure)
        try:
            # We need metadata to filter for positive class
            metadata_path = self.data_dir / 'stage2_train_metadata.csv'
            if not metadata_path.exists():
                print(f"Error: Metadata file not found at {metadata_path}. Cannot filter positive images.")
                return
                
            # Load and validate metadata
            metadata = pd.read_csv(metadata_path)
            
            # Check available columns
            print("Available columns:", metadata.columns.tolist())
            
            # Try different possible column names for patient ID
            patient_id_col = None
            for col in ['patientId', 'PatientId', 'patient_id', 'id']:
                if col in metadata.columns:
                    patient_id_col = col
                    break
            
            if not patient_id_col:
                print("Error: Could not find patient ID column in metadata.")
                print("Available columns:", metadata.columns.tolist())
                return

            # Try different possible column names and formats for target/label
            label_col = None
            positive_ids = []
            
            # Check for Target column with string labels
            if 'Target' in metadata.columns:
                if metadata['Target'].dtype == object:  # String labels
                    metadata['label'] = metadata['Target'].apply(lambda x: 1 if x == 'Lung Opacity' else 0)
                    positive_ids = metadata[metadata['label'] == 1][patient_id_col].tolist()
                else:  # Numeric labels
                    positive_ids = metadata[metadata['Target'] == 1][patient_id_col].tolist()
            # Check for direct label column
            elif 'label' in metadata.columns:
                positive_ids = metadata[metadata['label'] == 1][patient_id_col].tolist()
            # Check for class column
            elif 'class' in metadata.columns:
                metadata['label'] = metadata['class'].apply(lambda x: 1 if x == 'Lung Opacity' else 0)
                positive_ids = metadata[metadata['label'] == 1][patient_id_col].tolist()
            
            if not positive_ids:
                print("Error: No positive cases found in metadata.")
                print("Available columns:", metadata.columns.tolist())
                if 'Target' in metadata.columns:
                    print("Target value counts:", metadata['Target'].value_counts())
                elif 'label' in metadata.columns:
                    print("Label value counts:", metadata['label'].value_counts())
                elif 'class' in metadata.columns:
                    print("Class value counts:", metadata['class'].value_counts())
                return
            
            print(f"Found {len(positive_ids)} positive cases in metadata.")
            
            real_image_dir = self.data_dir / 'Training' / 'Images'
            if not real_image_dir.exists():
                print(f"Error: Real image directory not found at {real_image_dir}.")
                return
                
            # Check for image existence and collect valid paths
            real_paths = []
            for pid in positive_ids:
                img_path = real_image_dir / f"{pid}.png"
                if img_path.exists():
                    real_paths.append(img_path)
                else:
                    print(f"Warning: Image not found for patient {pid}")
            
            if not real_paths:
                print("Error: No positive real images found or path is incorrect.")
                print(f"Checked directory: {real_image_dir}")
                print(f"Example patient IDs: {positive_ids[:5]}")
                return
                
            print(f"Found {len(real_paths)} positive real images.")
            real_sample_paths = random.sample(real_paths, min(num_compare, len(real_paths)))
                
        except Exception as e:
            print(f"Error loading real positive images: {str(e)}")
            import traceback
            traceback.print_exc()
            return

        # Load synthetic images
        try:
            synthetic_paths = self._load_image_paths(self.synthetic_dir, num_compare)
            if not synthetic_paths:
                print(f"Error: No synthetic images found in {self.synthetic_dir}")
                return
            print(f"Found {len(synthetic_paths)} synthetic images.")
        except Exception as e:
            print(f"Error loading synthetic images: {str(e)}")
            import traceback
            traceback.print_exc()
            return

        # Ensure we have paths for comparison
        num_to_compare = min(len(real_sample_paths), len(synthetic_paths))
        if num_to_compare == 0:
            print("Error: Could not load sufficient images for comparison.")
            return
            
        real_sample_paths = real_sample_paths[:num_to_compare]
        synthetic_paths = synthetic_paths[:num_to_compare]

        ssim_scores = []
        print(f"Calculating SSIM for {num_to_compare} pairs...")
        for real_path, synth_path in zip(real_sample_paths, synthetic_paths):
            score = self._calculate_ssim(real_path, synth_path)
            if score is not None:
                ssim_scores.append(score)

        if not ssim_scores:
            print("SSIM calculation failed for all pairs.")
            return

        average_ssim = np.mean(ssim_scores)
        std_ssim = np.std(ssim_scores)

        # Report results
        ssim_report = (
            f"\n=== Synthetic Image Quality Analysis (SSIM) ===\n"
            f"Compared {len(ssim_scores)} pairs of real (positive class) vs. synthetic images.\n"
            f"  - Average SSIM: {average_ssim:.4f}\n"
            f"  - Std Dev SSIM: {std_ssim:.4f}\n"
            f"Images compared from:\n"
            f"  - Real: {self.data_dir / 'Training' / 'Images'}\n"
            f"  - Synthetic: {self.synthetic_dir}\n"
        )
        print(ssim_report)

        # Save SSIM results
        ssim_results_path = self.analysis_dir / 'ssim_analysis.json'
        results_data = {
            'num_pairs_compared': len(ssim_scores),
            'average_ssim': average_ssim,
            'std_dev_ssim': std_ssim,
            'individual_scores': ssim_scores # Optional: save individual scores
        }
        with open(ssim_results_path, 'w') as f:
            json.dump(results_data, f, indent=4)
        print(f"SSIM analysis results saved to {ssim_results_path}")

        # Optional: Save a few comparison images
        self._save_comparison_grid(real_sample_paths, synthetic_paths, num_examples=5)
        
        # Calculate FID and Precision/Recall
        try:
            # Get paths for real positive images
            real_paths = []
            metadata_path = self.data_dir / 'stage2_train_metadata.csv'
            metadata = pd.read_csv(metadata_path)
            
            # Get positive cases (reuse existing logic)
            positive_ids = []
            if 'Target' in metadata.columns:
                if metadata['Target'].dtype == object:
                    metadata['label'] = metadata['Target'].apply(lambda x: 1 if x == 'Lung Opacity' else 0)
                    positive_ids = metadata[metadata['label'] == 1]['patientId'].tolist()
                else:
                    positive_ids = metadata[metadata['Target'] == 1]['patientId'].tolist()
            
            real_image_dir = self.data_dir / 'Training' / 'Images'
            for pid in positive_ids:
                img_path = real_image_dir / f"{pid}.png"
                if img_path.exists():
                    real_paths.append(img_path)
            
            # Get synthetic image paths
            synthetic_paths = list(self.synthetic_dir.glob('*.png'))
            
            # Ensure we have enough images
            num_images = min(len(real_paths), len(synthetic_paths), num_compare)
            real_sample = random.sample(real_paths, num_images)
            synthetic_sample = random.sample(synthetic_paths, num_images)
            
            # Calculate FID
            fid_score = self.calculate_fid(real_sample, synthetic_sample)
            
            # Calculate features for precision/recall
            real_features = []
            synthetic_features = []
            
            with torch.no_grad():
                for paths, features_list in [(real_sample, real_features), (synthetic_sample, synthetic_features)]:
                    for path in tqdm(paths, desc="Extracting features"):
                        img = Image.open(path).convert('RGB')
                        img = self.fid_transform(img).unsqueeze(0)
                        if torch.cuda.is_available():
                            img = img.cuda()
                        features = self.inception(img)
                        features_list.append(features.cpu().numpy().flatten())
            
            real_features = np.array(real_features)
            synthetic_features = np.array(synthetic_features)
            
            # Calculate precision/recall
            precision, recall = self.calculate_precision_recall(real_features, synthetic_features)
            
            # Save comprehensive results
            results = {
                'fid_score': float(fid_score),
                'precision': float(precision),
                'recall': float(recall),
                'num_images_compared': num_images
            }
            
            # Add SSIM results if available
            if 'ssim_scores' in locals():
                results.update({
                    'average_ssim': float(np.mean(ssim_scores)),
                    'std_dev_ssim': float(np.std(ssim_scores))
                })
            
            # Save results
            results_path = self.analysis_dir / 'synthetic_quality_metrics.json'
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)
            
            # Print summary
            print("\n=== Synthetic Image Quality Analysis Results ===")
            print(f"Number of images compared: {num_images}")
            print(f"FID Score: {fid_score:.4f} (lower is better)")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            if 'ssim_scores' in locals():
                print(f"Average SSIM: {np.mean(ssim_scores):.4f}")
            
        except Exception as e:
            print(f"Error during quality metrics calculation: {str(e)}")
            import traceback
            traceback.print_exc()

    def _save_comparison_grid(self, real_paths: List[Path], synth_paths: List[Path], num_examples: int = 5):
        """Save a grid comparing real and synthetic images."""
        num_examples = min(num_examples, len(real_paths), len(synth_paths))
        if num_examples == 0:
            return
            
        fig, axes = plt.subplots(num_examples, 2, figsize=(6, 3 * num_examples))
        fig.suptitle("Real vs. Synthetic Image Examples", fontsize=14)
        
        for i in range(num_examples):
            real_img = Image.open(real_paths[i]).convert('L') # Load as grayscale
            synth_img = Image.open(synth_paths[i]).convert('L')
            
            ax_real = axes[i, 0]
            ax_synth = axes[i, 1]
            
            ax_real.imshow(real_img, cmap='gray')
            ax_real.set_title(f"Real: {real_paths[i].name}")
            ax_real.axis('off')
            
            ax_synth.imshow(synth_img, cmap='gray')
            ax_synth.set_title(f"Synthetic: {synth_paths[i].name}")
            ax_synth.axis('off')
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap
        save_path = self.analysis_dir / 'real_vs_synthetic_comparison.png'
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Comparison image grid saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze and compare baseline and augmented model results, including SSIM analysis')
    parser.add_argument('--metrics-dir', type=str, default='./results/metrics',
                        help='Directory containing metrics files')
    parser.add_argument('--analysis-dir', type=str, default='./results/analysis',
                        help='Directory to save analysis outputs (figures, reports)')
    # Arguments for SSIM analysis
    parser.add_argument('--data-dir', type=str, default='./data/processed',
                        help='Path to the processed real dataset directory (needed for SSIM)')
    parser.add_argument('--synthetic-dir', type=str, default='./data/synthetic',
                        help='Path to the synthetic image directory (needed for SSIM)')
    parser.add_argument('--num-compare-images', type=int, default=100,
                        help='Number of real/synthetic image pairs to compare for SSIM')
    parser.add_argument('--skip-ssim', action='store_true',
                        help='Skip the SSIM analysis step')

    args = parser.parse_args()
    
    analyzer = ResultsAnalyzer(
        metrics_dir=args.metrics_dir, 
        analysis_dir=args.analysis_dir,
        data_dir=args.data_dir,
        synthetic_dir=args.synthetic_dir
    )
    
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

    # --- Analyze Synthetic Image Quality (SSIM) --- #
    if not args.skip_ssim:
        analyzer.analyze_synthetic_quality(num_compare=args.num_compare_images)
    else:
        print("\nSkipping SSIM analysis as requested.")

    print(f"\nAnalysis complete. Results saved to {args.analysis_dir}")

if __name__ == '__main__':
    main() 