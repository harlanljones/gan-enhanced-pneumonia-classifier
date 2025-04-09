import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import os
import argparse
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Project specific imports
from data_loader import (get_dataloaders, get_kfold_dataloaders,
                         get_augmented_dataloaders, get_augmented_kfold_dataloaders)
from classifier import create_resnet50_baseline

def train_model(model, criterion, optimizer, dataloaders, device, num_epochs=25, scheduler=None,
                model_save_path='../models', results_save_path='../results/metrics',
                fold=None, use_synthetic=False):
    """
    Trains and validates the model.

    Args:
        model (nn.Module): The model to train.
        criterion: The loss function.
        optimizer: The optimizer.
        dataloaders (dict): Dictionary containing 'train' and 'val' DataLoaders.
        device: The device (CPU or CUDA).
        num_epochs (int): Number of training epochs.
        scheduler: Learning rate scheduler (optional).
        model_save_path (str): Directory to save model checkpoints.
        results_save_path (str): Directory to save training metrics.
        fold (int, optional): Current fold number if using K-Fold CV.
        use_synthetic (bool): Whether synthetic data was used (for naming output files).

    Returns:
        tuple: (best_model_state, history)
               best_model_state: State dictionary of the best performing model based on validation accuracy.
               history: Dictionary containing training and validation loss/accuracy history.
    """
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(results_save_path, exist_ok=True)

    fold_prefix = f"fold_{fold}_" if fold is not None else ""
    run_prefix = f"{fold_prefix}{'augmented' if use_synthetic else 'baseline'}_"

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            all_preds = []
            all_labels = []

            # Use tqdm for progress bar
            progress_bar = tqdm(dataloaders[phase], desc=f'{phase.capitalize()} Epoch {epoch+1}', leave=False)
            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Update progress bar
                progress_bar.set_postfix(loss=loss.item())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = accuracy_score(all_labels, all_preds)

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it's the best validation accuracy so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                # Save the best model checkpoint for this fold/run
                best_model_filename = os.path.join(model_save_path, f'{run_prefix}resnet50.pth')
                torch.save(best_model_wts, best_model_filename)
                print(f"Saved best model checkpoint to {best_model_filename}")

        if scheduler and phase == 'train': # Scheduler step usually after training phase or validation phase
             # Example: scheduler.step() or scheduler.step(epoch_loss) if ReduceLROnPlateau
             scheduler.step() # Assuming StepLR or similar

        epoch_time = time.time() - epoch_start
        print(f"Epoch completed in {epoch_time // 60:.0f}m {epoch_time % 60:.0f}s")
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Save training history
    history_filename = os.path.join(results_save_path, f'{run_prefix}training_history.json')
    with open(history_filename, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Saved training history to {history_filename}")

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

def evaluate_model(model, dataloader, device, criterion):
    """
    Evaluates the model on a given dataset.

    Args:
        model (nn.Module): The trained model to evaluate.
        dataloader (DataLoader): DataLoader for the evaluation set.
        device: The device (CPU or CUDA).
        criterion: The loss function.

    Returns:
        dict: Dictionary containing evaluation metrics (loss, accuracy, precision, recall, f1).
    """
    model.eval() # Set model to evaluate mode
    running_loss = 0.0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc='Evaluating', leave=False)
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    eval_loss = running_loss / len(dataloader.dataset)
    eval_accuracy = accuracy_score(all_labels, all_preds)
    # Calculate weighted metrics to account for class imbalance
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    # Keep binary metrics if needed for specific positive class analysis (optional)
    # precision_b, recall_b, f1_b, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)

    metrics = {
        'loss': eval_loss,
        'accuracy': eval_accuracy,
        'weighted_precision': precision_w,
        'weighted_recall': recall_w,
        'weighted_f1_score': f1_w
        # 'binary_precision': precision_b, # Uncomment if needed
        # 'binary_recall': recall_b,     # Uncomment if needed
        # 'binary_f1_score': f1_b        # Uncomment if needed
    }

    print(f"Evaluation Results - Loss: {eval_loss:.4f}, Accuracy: {eval_accuracy:.4f}, Weighted Precision: {precision_w:.4f}, Weighted Recall: {recall_w:.4f}, Weighted F1: {f1_w:.4f}")
    return metrics

# --- Plotting Functions Integrated ---

def load_history(filepath, run_prefix):
    """
    Loads training history from a JSON file, prepending run_prefix to filename.

    Args:
        filepath (str): Base directory containing the JSON file.
        run_prefix (str): Prefix for the history filename (e.g., "baseline_", "augmented_", "fold_0_baseline_").

    Returns:
        dict or None: Loaded history dictionary or None if an error occurs.
    """
    history_filename = os.path.join(filepath, f"{run_prefix}training_history.json")
    try:
        with open(history_filename, 'r') as f:
            history = json.load(f)
        # Ensure all lists have the same length, padding if necessary
        max_len = 0
        valid_keys = [k for k, v in history.items() if isinstance(v, list)]
        for key in valid_keys:
            max_len = max(max_len, len(history[key]))
        for key in valid_keys:
            current_len = len(history[key])
            if current_len < max_len:
                padding_value = history[key][-1] if current_len > 0 else np.nan
                history[key].extend([padding_value] * (max_len - current_len))
        return history
    except FileNotFoundError:
        print(f"Warning: History file not found: {history_filename}")
        return None
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from: {history_filename}")
        return None
    except Exception as e:
        print(f"Warning: An error occurred loading {history_filename}: {e}")
        return None

def plot_metric(histories, metric_key_base, title, ylabel, output_path, run_prefix=""):
    """Plots a specific metric (loss or accuracy) for multiple training histories, optionally adding average lines for CV."""
    plt.figure(figsize=(12, 7))
    is_cv = len(histories) > 1
    all_train_metrics = []
    all_val_metrics = []
    max_epochs = 0

    # Color scheme
    train_color = '#1f77b4'  # Blue
    val_color = '#ff7f0e'    # Orange
    avg_train_color = '#2ca02c'  # Green
    avg_val_color = '#d62728'    # Red

    for i, history in enumerate(histories):
        if history:
            train_key = f'train_{metric_key_base}'
            val_key = f'val_{metric_key_base}'
            
            if train_key in history and val_key in history and len(history[train_key]) > 0:
                epochs = range(1, len(history[train_key]) + 1)
                max_epochs = max(max_epochs, len(epochs))
                
                if is_cv:
                    all_train_metrics.append(history[train_key])
                    all_val_metrics.append(history[val_key])

                # Plot individual lines with transparency for CV
                alpha = 0.3 if is_cv else 1.0
                plt.plot(epochs, history[train_key], 
                        color=train_color, linestyle='-', alpha=alpha,
                        label=f'{"Fold" if is_cv else "Run"} {i+1} Train')
                plt.plot(epochs, history[val_key], 
                        color=val_color, linestyle='--', alpha=alpha,
                        label=f'{"Fold" if is_cv else "Run"} {i+1} Val')
            else:
                print(f"Warning: Missing keys '{train_key}' or '{val_key}' in history {i+1}")

    # Add average lines if CV and data exists
    if is_cv and all_train_metrics and all_val_metrics:
        try:
            max_len = max(len(m) for m in all_train_metrics + all_val_metrics)
            padded_train = [m + [m[-1]]*(max_len - len(m)) for m in all_train_metrics]
            padded_val = [m + [m[-1]]*(max_len - len(m)) for m in all_val_metrics]

            mean_train = np.nanmean(np.array(padded_train), axis=0)
            mean_val = np.nanmean(np.array(padded_val), axis=0)
            std_train = np.nanstd(np.array(padded_train), axis=0)
            std_val = np.nanstd(np.array(padded_val), axis=0)

            epochs = range(1, max_len + 1)
            
            # Plot mean lines
            plt.plot(epochs, mean_train, color=avg_train_color, 
                    linestyle='-', linewidth=2, label='Average Train')
            plt.plot(epochs, mean_val, color=avg_val_color, 
                    linestyle='--', linewidth=2, label='Average Val')

            # Add confidence intervals
            plt.fill_between(epochs, 
                           mean_train - std_train, 
                           mean_train + std_train, 
                           color=avg_train_color, alpha=0.1)
            plt.fill_between(epochs, 
                           mean_val - std_val, 
                           mean_val + std_val, 
                           color=avg_val_color, alpha=0.1)
        except Exception as e:
            print(f"Warning: Could not calculate or plot average lines: {e}")

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    
    if max_epochs > 0:
        plt.xlim(1, max_epochs)
        if 'acc' in metric_key_base.lower():
            plt.ylim(0, 1.05)
    
    # Always place legend outside if CV or if there are averages
    if is_cv or (is_cv and len(histories) > 1): # simplified condition
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    elif len(histories) == 1:
        plt.legend(loc='best')
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.85, 1] if is_cv else [0, 0, 1, 1]) # Adjust layout for external legend
    plt.savefig(os.path.join(output_path, f"{run_prefix}{metric_key_base}_curves.png"))
    plt.close()
    print(f"Saved {metric_key_base} plot to {output_path}")


def plot_cv_summary(cv_summary_path, output_dir, run_prefix):
    """Plots the summary of cross-validation results from a JSON file."""
    try:
        summary_filename = os.path.join(cv_summary_path, f"{run_prefix}cv_summary.json")
        with open(summary_filename, 'r') as f:
            cv_results = json.load(f)
    except FileNotFoundError:
        print(f"Warning: CV summary file not found: {summary_filename}. Cannot plot CV summary.")
        return
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from: {summary_filename}")
        return

    if 'folds' not in cv_results or 'average' not in cv_results:
        print("Warning: No valid 'folds' or 'average' key found in CV summary.")
        return

    # Define the metrics we want to plot (excluding loss which will be plotted separately)
    metric_mapping = {
        'accuracy': 'Accuracy',
        'weighted_precision': 'Precision',
        'weighted_recall': 'Recall',
        'weighted_f1_score': 'F1 score'
    }

    num_folds = len(cv_results['folds'])
    folds = [f"Fold {i+1}" for i in range(num_folds)]
    index = np.arange(num_folds)

    # --- Plot 1: Primary Metrics (Accuracy, Precision, Recall, F1) ---
    plt.figure(figsize=(12, 7))
    bar_width = 0.2  # Width of each bar
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Different color for each metric

    for i, (metric_key, metric_label) in enumerate(metric_mapping.items()):
        values = [fold_metrics.get(metric_key, 0.0) for fold_metrics in cv_results['folds']]
        
        # Plot bars for this metric
        bars = plt.bar(index + i * bar_width - (len(metric_mapping)-1) * bar_width/2,
                      values,
                      bar_width,
                      label=metric_label,
                      color=colors[i],
                      alpha=0.8)

        # Add average line for this metric
        if metric_key in cv_results['average']:
            avg_value = cv_results['average'][metric_key]
            plt.hlines(avg_value,
                      xmin=index[0] - bar_width,
                      xmax=index[-1] + bar_width * len(metric_mapping),
                      colors=colors[i],
                      linestyles='dashed',
                      label=f'Avg {metric_label}: {avg_value:.3f}')

    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title('Test Set Performance Metrics per Fold (Cross-Validation)')
    plt.xticks(index, folds)
    plt.ylim(0, 1.05)
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{run_prefix}cv_test_metrics_per_fold.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics plot to {output_path}")

    # --- Plot 2: Loss ---
    if any('loss' in fold_metrics for fold_metrics in cv_results['folds']):
        plt.figure(figsize=(10, 6))
        loss_values = [fold_metrics.get('loss', np.nan) for fold_metrics in cv_results['folds']]

        if not all(np.isnan(loss_values)):
            plt.bar(index, loss_values, color='#1f77b4', alpha=0.8, label='Test Loss')

            # Plot average loss line
            if 'loss' in cv_results['average']:
                avg_loss = cv_results['average']['loss']
                plt.hlines(avg_loss,
                          xmin=index[0]-0.5,
                          xmax=index[-1]+0.5,
                          colors='red',
                          linestyles='dashed',
                          label=f'Avg Loss: {avg_loss:.4f}')

            plt.xlabel('Fold')
            plt.ylabel('Loss')
            plt.title('Test Set Loss per Fold (Cross-Validation)')
            plt.xticks(index, folds)
            min_loss = min(v for v in loss_values if not np.isnan(v))
            max_loss = max(v for v in loss_values if not np.isnan(v))
            plt.ylim(min_loss * 0.9, max_loss * 1.1)
            plt.legend(loc='upper right')
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            output_path = os.path.join(output_dir, f'{run_prefix}cv_test_loss_per_fold.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved loss plot to {output_path}")
        else:
            plt.close()
            print("Skipping CV loss plot as no valid loss data was found.")

    print(f"Saved CV summary plot to {os.path.join(output_dir, f'{run_prefix}cv_metrics_summary.png')}")


def generate_plots(metrics_dir, figures_dir, use_synthetic=False, k_folds=5):
    """
    Generates loss and accuracy plots from saved training history files.
    If k_folds > 0, generates plots for each fold and an average plot, plus a CV summary plot.
    If k_folds == 0, generates a single plot for the standard train/test run.

    Args:
        metrics_dir (str): Directory containing the training history JSON files.
        figures_dir (str): Directory to save the generated plots.
        use_synthetic (bool): Whether synthetic data was used (for file naming).
        k_folds (int): Number of folds used during training (0 for no CV).
    """
    os.makedirs(figures_dir, exist_ok=True)
    print(f"\nGenerating plots in {figures_dir}...")

    run_prefix_base = "augmented_" if use_synthetic else "baseline_"

    if k_folds > 0:
        fold_histories = []
        for i in range(k_folds):
            fold_run_prefix = f"fold_{i}_{run_prefix_base}"
            history = load_history(metrics_dir, fold_run_prefix)
            if history:
                fold_histories.append(history)
            else:
                print(f"Warning: Could not load history for fold {i} with prefix {fold_run_prefix}")

        if fold_histories:
            print(f"Plotting average metrics across {len(fold_histories)} folds...")
            plot_metric(fold_histories, 'loss', f'Avg Train/Val Loss ({run_prefix_base[:-1]}, CV)', 'Loss', figures_dir, run_prefix=f"avg_{run_prefix_base}")
            plot_metric(fold_histories, 'acc', f'Avg Train/Val Accuracy ({run_prefix_base[:-1]}, CV)', 'Accuracy', figures_dir, run_prefix=f"avg_{run_prefix_base}")

            # Generate CV summary plot
            plot_cv_summary(metrics_dir, figures_dir, run_prefix=run_prefix_base)
        else:
            print("No valid fold histories found to generate average plots.")

    else: # No cross-validation (k_folds == 0)
        history = load_history(metrics_dir, run_prefix_base)
        if history:
            print(f"Plotting metrics for single run ({run_prefix_base[:-1]})...")
            plot_metric([history], 'loss', f'Train/Val Loss ({run_prefix_base[:-1]})', 'Loss', figures_dir, run_prefix=run_prefix_base)
            plot_metric([history], 'acc', f'Train/Val Accuracy ({run_prefix_base[:-1]})', 'Accuracy', figures_dir, run_prefix=run_prefix_base)
        else:
            print(f"No history file found for prefix {run_prefix_base} in {metrics_dir} for single run.")

    print(f"--- Plot Generation Complete ---")


# --- End of Plotting Functions ---


def main(args):
    """Main function to orchestrate training, evaluation, and plotting.
    
    Supports both standard training and training with synthetic data augmentation.
    When using synthetic data (--use-synthetic), the training set is augmented with
    generated images while validation and test sets remain unchanged to ensure fair evaluation.
    """
    # Setup device
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure results and model directories exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # Print run configuration
    print("\n=== Training Configuration ===")
    print(f"{'Using synthetic data:':25} {'Yes' if args.use_synthetic else 'No'}")
    if args.use_synthetic:
        print(f"{'Synthetic data directory:':25} {args.synthetic_dir}")
    print(f"{'Cross-validation:':25} {'Yes (' + str(args.k_folds) + ' folds)' if args.k_folds > 0 else 'No'}")
    print(f"{'Base model:':25} ResNet50 ({'fine-tuned' if args.unfreeze else 'feature extraction'})")
    print(f"{'Training epochs:':25} {args.epochs}")
    print(f"{'Batch size:':25} {args.batch_size}")
    print(f"{'Learning rate:':25} {args.lr}")
    print("="*30 + "\n")

    criterion = nn.CrossEntropyLoss()

    # Load data with appropriate loader based on configuration
    print("\n=== Loading Dataset ===")
    try:
        if args.k_folds > 0:
            if args.use_synthetic:
                print("Loading data for cross-validation with synthetic augmentation...")
                fold_dataloaders, test_loader = get_augmented_kfold_dataloaders(
                    data_dir=args.data_dir,
                    synthetic_dir=args.synthetic_dir,
                    k_folds=args.k_folds,
                    batch_size=args.batch_size,
                    num_workers=args.workers
                )
            else:
                print("Loading data for standard cross-validation...")
                fold_dataloaders, test_loader = get_kfold_dataloaders(
                    data_dir=args.data_dir,
                    k_folds=args.k_folds,
                    batch_size=args.batch_size,
                    num_workers=args.workers
                )
        else:
            if args.use_synthetic:
                print("Loading data for single run with synthetic augmentation...")
                train_loader, test_loader = get_augmented_dataloaders(
                    data_dir=args.data_dir,
                    synthetic_dir=args.synthetic_dir,
                    batch_size=args.batch_size,
                    num_workers=args.workers
                )
            else:
                print("Loading data for standard single run...")
                train_loader, test_loader = get_dataloaders(
                    data_dir=args.data_dir,
                    batch_size=args.batch_size,
                    num_workers=args.workers
                )
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure both the original dataset and (if using) synthetic images are available.")
        return
    except Exception as e:
        print(f"\nError loading data: {e}")
        return

    print("Data loading completed successfully.")
    print("="*30 + "\n")

    if args.k_folds > 0:
        print(f"\n=== Starting {args.k_folds}-Fold Cross Validation ===")
        if test_loader is None:
            print("ERROR: Test loader could not be created. Check test data directory and structure.")
            return

        all_fold_histories = []
        test_metrics_per_fold = []
        best_fold_val_acc = -1
        best_model_state_overall = None

        for i, fold_data in enumerate(fold_dataloaders):
            print(f"\n{'='*20} Fold {i+1}/{args.k_folds} {'='*20}")
            
            # Create a new model instance for each fold
            model = create_resnet50_baseline(num_classes=2, pretrained=True, freeze_base=not args.unfreeze)
            model = model.to(device)

            print(f"Training fold {i+1} with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

            # Setup optimizer
            params_to_optimize = model.parameters() if args.unfreeze else model.fc.parameters()
            optimizer = optim.Adam(params_to_optimize, lr=args.lr)
            scheduler = None

            # Train the model for this fold
            fold_model, fold_history = train_model(
                model, criterion, optimizer, fold_data, device, args.epochs, scheduler,
                args.model_dir, args.results_dir, fold=i+1, use_synthetic=args.use_synthetic
            )
            all_fold_histories.append(fold_history)

            # Evaluate on test set
            print(f"\n--- Evaluating Fold {i+1} Best Model on Test Set ---")
            fold_test_metrics = evaluate_model(fold_model, test_loader, device, criterion)
            test_metrics_per_fold.append(fold_test_metrics)

            # Track best model
            current_best_val_acc = max(fold_history['val_acc'])
            if current_best_val_acc > best_fold_val_acc:
                best_fold_val_acc = current_best_val_acc
                best_model_state_overall = fold_model.state_dict()
                model_prefix = 'augmented' if args.use_synthetic else 'baseline'
                overall_best_path = os.path.join(args.model_dir, f'best_overall_{model_prefix}_resnet50.pth')
                torch.save(best_model_state_overall, overall_best_path)
                print(f"New best model (val_acc={best_fold_val_acc:.4f}) from fold {i+1}")
                print(f"Saved to {overall_best_path}")

        # Print cross-validation summary
        print("\n=== Cross-Validation Summary ===")
        avg_test_metrics = {
            key: np.mean([m[key] for m in test_metrics_per_fold])
            for key in test_metrics_per_fold[0].keys()
        }
        std_test_metrics = {
            key: np.std([m[key] for m in test_metrics_per_fold])
            for key in test_metrics_per_fold[0].keys()
        }

        print("\nAverage Test Metrics across folds:")
        for metric in ['loss', 'accuracy', 'weighted_precision', 'weighted_recall', 'weighted_f1_score']:
            print(f"  {metric.replace('_', ' ').title():20}: {avg_test_metrics[metric]:.4f} ± {std_test_metrics[metric]:.4f}")

        # Save CV summary
        run_prefix = "augmented_" if args.use_synthetic else "baseline_"
        cv_results_path = os.path.join(args.results_dir, f'{run_prefix}cv_summary.json')
        summary_data = {
            'average': avg_test_metrics,
            'std_dev': std_test_metrics,
            'folds': test_metrics_per_fold,
            'config': vars(args)
        }
        with open(cv_results_path, 'w') as f:
            json.dump(summary_data, f, indent=4)
        print(f"\nSaved detailed CV summary to {cv_results_path}")

    else:
        print("\n=== Starting Single Train/Test Run ===")
        if test_loader is None:
            print("ERROR: Test loader could not be created. Check test data directory and structure.")
            return

        model = create_resnet50_baseline(num_classes=2, pretrained=True, freeze_base=not args.unfreeze)
        model = model.to(device)

        print(f"Training model with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

        params_to_optimize = model.parameters() if args.unfreeze else model.fc.parameters()
        optimizer = optim.Adam(params_to_optimize, lr=args.lr)
        scheduler = None

        dataloaders = {'train': train_loader, 'val': test_loader}

        best_model, history = train_model(
            model, criterion, optimizer, dataloaders, device, args.epochs, scheduler,
            args.model_dir, args.results_dir, fold=None, use_synthetic=args.use_synthetic
        )

        print("\n=== Final Evaluation on Test Set ===")
        final_metrics = evaluate_model(best_model, test_loader, device, criterion)

        # Save final metrics with configuration
        run_prefix = "augmented_" if args.use_synthetic else "baseline_"
        final_metrics_path = os.path.join(args.results_dir, f'{run_prefix}final_metrics.json')
        final_results = {
            'metrics': final_metrics,
            'config': vars(args)
        }
        with open(final_metrics_path, 'w') as f:
            json.dump(final_results, f, indent=4)
        print(f"\nSaved final evaluation results to {final_metrics_path}")

    # Generate plots
    print("\n=== Generating Performance Plots ===")
    try:
        generate_plots(args.results_dir, args.figures_dir, args.use_synthetic, args.k_folds)
        print("Plot generation completed successfully.")
    except ImportError:
        print("\nWarning: Matplotlib not found. Skipping plot generation.")
        print("Install matplotlib to generate plots: pip install matplotlib")
    except Exception as e:
        print(f"\nError during plot generation: {e}")

    print("\n=== Training Complete ===")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    Train ResNet50 Classifier for Pneumonia Detection
    
    This script supports both standard training and training with synthetic data augmentation.
    When using synthetic data (--use-synthetic), the training set is augmented while validation
    and test sets remain unchanged to ensure fair evaluation.
    ''')
    
    # Data and output directories
    parser.add_argument('--data-dir', type=str, default='./data/processed',
                        help='Path to the processed dataset directory')
    parser.add_argument('--model-dir', type=str, default='./models',
                        help='Directory to save model checkpoints')
    parser.add_argument('--results-dir', type=str, default='./results/metrics',
                        help='Directory to save metrics and evaluation results')
    parser.add_argument('--figures-dir', type=str, default='./results/figures',
                        help='Directory to save performance plots')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--unfreeze', action='store_true',
                        help='Unfreeze and fine-tune all ResNet layers (default: only train FC layer)')
    
    # Synthetic data options
    parser.add_argument('--use-synthetic', action='store_true',
                        help='Augment training data with synthetic images')
    parser.add_argument('--synthetic-dir', type=str, default='./data/synthetic',
                        help='Directory containing synthetic images (used with --use-synthetic)')
    
    # Cross-validation options
    parser.add_argument('--k-folds', type=int, default=5,
                        help='Number of folds for cross-validation (0 to disable)')
    
    # Hardware options
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage even if CUDA is available')

    args = parser.parse_args()

    # Validate k-folds argument
    if args.k_folds <= 1:
        args.k_folds = 0
        print("Note: k_folds <= 1, disabling cross-validation")

    # Check dependencies
    try:
        import matplotlib.pyplot as plt
        import numpy
    except ImportError as e:
        print(f"Warning: Missing plotting dependency ({e})")
        print("Install required packages: pip install matplotlib numpy")

    main(args)
