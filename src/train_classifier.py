import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
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

from data_loader import (
    get_dataloaders, get_kfold_dataloaders,
    get_simple_augmented_dataloaders, get_simple_augmented_kfold_dataloaders, 
    get_phased_augmented_kfold_dataloaders, 
    PhasedAugmentedDataset 
)
from classifier import create_resnet50_baseline
from utils import check_create_dir

# --- Curriculum Schedule Parsing ---
def parse_curriculum_schedule(schedule_str: str) -> dict:
    """Parses a curriculum schedule string (e.g., "0:0.0, 5:0.25, 10:0.5") into a dictionary."""
    schedule = {}
    if not schedule_str:
        return schedule
    try:
        parts = schedule_str.split(',')
        for part in parts:
            epoch_str, ratio_str = part.strip().split(':')
            epoch = int(epoch_str)
            ratio = float(ratio_str)
            if not (0 <= epoch):
                 raise ValueError(f"Epoch must be non-negative: {epoch}")
            if not (0.0 <= ratio <= 1.0):
                raise ValueError(f"Ratio must be between 0.0 and 1.0: {ratio}")
            schedule[epoch] = ratio
        schedule = dict(sorted(schedule.items()))
        if 0 not in schedule:
            schedule[0] = 0.0
            schedule = dict(sorted(schedule.items())) 
        return schedule
    except Exception as e:
        raise ValueError(f"Invalid curriculum schedule format: '{schedule_str}'. Error: {e}. Expected format: 'epoch1:ratio1, epoch2:ratio2,...'")

def get_current_synthetic_ratio(epoch: int, schedule: dict) -> float:
    """Gets the synthetic ratio for the current epoch based on the schedule."""
    if not schedule:
        return 0.0 # Default to no synthetic data if no schedule

    current_ratio = 0.0
    applicable_epochs = [e for e in schedule.keys() if e <= epoch]
    if applicable_epochs:
        current_ratio = schedule[max(applicable_epochs)]
    elif 0 in schedule:
        current_ratio = schedule[0]

    return current_ratio

# --- Modified Training Function ---
def train_model(model, criterion, optimizer, dataloaders, device, num_epochs=25, scheduler=None,
                model_save_path='../models', results_save_path='../results/metrics',
                fold=None, use_synthetic=False, curriculum_schedule=None):
    """
    Trains and validates the model, potentially using phased curriculum augmentation.

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
        use_synthetic (bool): Indicates if *any* synthetic data is used (for naming/logic).
        curriculum_schedule (dict, optional): Parsed schedule for phased augmentation {epoch: ratio}.
                                           If None or empty, uses fixed augmentation if use_synthetic=True.

    Returns:
        tuple: (best_model_state, history)
               best_model_state: State dictionary of the best performing model based on validation accuracy.
               history: Dictionary containing training and validation loss/accuracy history.
    """
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    history = {
        'epoch': [],
        'train_loss': [], 
        'train_acc': [], 
        'val_loss': [], 
        'val_acc': [], 
        'synthetic_ratio': []
    }

    check_create_dir(model_save_path)
    check_create_dir(results_save_path)

    fold_prefix = f"fold_{fold}_" if fold is not None else ""
    aug_type = "curriculum" if use_synthetic and curriculum_schedule else ("augmented" if use_synthetic else "baseline")
    run_prefix = f"{fold_prefix}{aug_type}_"

    train_dataset = None
    if use_synthetic and curriculum_schedule and isinstance(dataloaders['train'], DataLoader) and hasattr(dataloaders['train'].dataset, 'set_synthetic_ratio'):
        train_dataset = dataloaders['train'].dataset
        print("Phased curriculum augmentation enabled.")
    elif use_synthetic:
        print("Simple concatenation augmentation enabled.")
    else:
        print("Baseline training (no synthetic data).")

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        current_ratio = 0.0
        if train_dataset and curriculum_schedule:
            current_ratio = get_current_synthetic_ratio(epoch, curriculum_schedule)
            train_dataset.set_synthetic_ratio(current_ratio)
        elif use_synthetic and not curriculum_schedule:
            current_ratio = 1.0 if isinstance(dataloaders['train'].dataset, torch.utils.data.ConcatDataset) else 0.0

        history['epoch'].append(epoch + 1) 
        history['synthetic_ratio'].append(current_ratio)
        print(f"Current Synthetic Ratio: {current_ratio:.2f}")

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() 
            else:
                model.eval()

            running_loss = 0.0
            all_preds = []
            all_labels = []

            data_loader = dataloaders[phase]
            progress_bar = tqdm(data_loader, desc=f'{phase.capitalize()} Epoch {epoch+1}', leave=False)

            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                progress_bar.set_postfix(loss=loss.item())

            epoch_samples = len(data_loader.dataset)
            epoch_loss = running_loss / epoch_samples
            epoch_acc = accuracy_score(all_labels, all_preds)

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                best_model_filename = os.path.join(model_save_path, f'{run_prefix}resnet50.pth')
                torch.save(best_model_wts, best_model_filename)
                print(f"Saved best model checkpoint to {best_model_filename}")

        if scheduler and phase == 'train':
             scheduler.step()

        epoch_time = time.time() - epoch_start
        print(f"Epoch completed in {epoch_time // 60:.0f}m {epoch_time % 60:.0f}s")
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    history_filename = os.path.join(results_save_path, f'{run_prefix}training_history.json')
    with open(history_filename, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Saved training history to {history_filename}")

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
    model.eval()
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
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)

    metrics = {
        'loss': eval_loss,
        'accuracy': eval_accuracy,
        'weighted_precision': precision_w,
        'weighted_recall': recall_w,
        'weighted_f1_score': f1_w
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

def plot_metric(histories, metric_key, title, ylabel, output_path, run_prefix="", plot_ratio=False):
    """Plot training and validation metrics."""
    plt.figure(figsize=(10, 6))
    
    train_metrics = []
    val_metrics = []
    
    for history in histories:
        if plot_ratio:
            if 'synthetic_ratio' not in history:
                continue
            values = history['synthetic_ratio']
            epochs = range(1, len(values) + 1)
            plt.plot(epochs, values, alpha=0.3, color='blue',
                    label=f'Fold {history.get("fold", "")}' if 'fold' in history else 'Ratio')
            train_metrics.append(values)
        else:
            if metric_key not in history:
                continue
            
            train_values = history[metric_key]
            val_key = f'val_{metric_key.split("train_")[1]}' if metric_key.startswith('train_') else f'val_{metric_key}'
            val_values = history.get(val_key, [])
            
            epochs = range(1, len(train_values) + 1)
            
            plt.plot(epochs, train_values, alpha=0.3, color='blue',
                    label=f'Train Fold {history.get("fold", "")}' if 'fold' in history else 'Training')
            if val_values:
                plt.plot(epochs, val_values, alpha=0.3, color='orange',
                        label=f'Val Fold {history.get("fold", "")}' if 'fold' in history else 'Validation')
            
            train_metrics.append(train_values)
            if val_values:
                val_metrics.append(val_values)
    
    if not train_metrics:
        raise ValueError(f"No valid data found for metric: {metric_key}")
    
    epochs = range(1, len(train_metrics[0]) + 1) 
    
    if plot_ratio:
        ratio_avg = np.mean(train_metrics, axis=0)
        plt.plot(epochs, ratio_avg, 'b-', label='Average Ratio', linewidth=2)
    else:
        train_avg = np.mean(train_metrics, axis=0)
        plt.plot(epochs, train_avg, 'b-', label='Average Training', linewidth=2)
        
        if val_metrics:
            val_avg = np.mean(val_metrics, axis=0)
            plt.plot(epochs, val_avg, 'orange', label='Average Validation', linewidth=2)
    
    plt.title(f'{title} - {run_prefix}' if run_prefix else title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

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
    bar_width = 0.2  
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 

    for i, (metric_key, metric_label) in enumerate(metric_mapping.items()):
        values = [fold_metrics.get(metric_key, 0.0) for fold_metrics in cv_results['folds']]
        
        bars = plt.bar(index + i * bar_width - (len(metric_mapping)-1) * bar_width/2,
                      values,
                      bar_width,
                      label=metric_label,
                      color=colors[i],
                      alpha=0.8)

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


def generate_plots(metrics_dir, figures_dir, run_prefix="", k_folds=None):
    """Generates training plots for a given run (baseline, augmented, curriculum) or CV folds."""
    check_create_dir(figures_dir)
    histories = []

    if k_folds and k_folds > 1:
        print(f"Generating plots for {k_folds}-Fold CV run: {run_prefix}...")
        for fold in range(1, k_folds + 1): 
            fold_run_prefix = f"fold_{fold}_{run_prefix}"
            history = load_history(metrics_dir, fold_run_prefix)
            if history:
                history['fold'] = fold
                histories.append(history)
        if not histories:
            print(f"No history files found for CV run prefix: {run_prefix}")
            return
    else:
        print(f"Generating plots for single run: {run_prefix}...")
        history = load_history(metrics_dir, run_prefix)
        if history:
            histories.append(history)
        else:
            print(f"No history file found for run prefix: {run_prefix}")
            return

    # Plot Loss
    try:
        plot_metric(histories, 'train_loss', 'Training Loss', 'Loss',
                    os.path.join(figures_dir, f'{run_prefix}loss_curve.png'), run_prefix)
    except Exception as e:
        print(f"Warning: Could not generate loss plot: {e}")

    # Plot Accuracy
    try:
        plot_metric(histories, 'train_acc', 'Training Accuracy', 'Accuracy',
                    os.path.join(figures_dir, f'{run_prefix}accuracy_curve.png'), run_prefix)
    except Exception as e:
        print(f"Warning: Could not generate accuracy plot: {e}")

    # Plot Synthetic Ratio if present
    if histories and any('synthetic_ratio' in h for h in histories):
        try:
            plot_metric(histories, 'synthetic_ratio', 'Synthetic Data Ratio', 'Ratio',
                       os.path.join(figures_dir, f'{run_prefix}synthetic_ratio_curve.png'),
                       run_prefix, plot_ratio=True)
        except Exception as e:
            print(f"Warning: Could not generate synthetic ratio plot: {e}")

    # Plot CV Summary Bar Chart if applicable
    if k_folds and k_folds > 1:
        cv_summary_path = os.path.join(metrics_dir, f"{run_prefix}cv_summary.json")
        if os.path.exists(cv_summary_path):
            try:
                plot_cv_summary(metrics_dir, figures_dir, run_prefix)  # Pass metrics_dir instead of cv_summary_path
            except Exception as e:
                print(f"Warning: Could not generate CV summary plots: {e}")
        else:
            print(f"CV Summary file not found: {cv_summary_path}")


# --- Modified Main Function ---
def main(args):
    print("Starting Classifier Training...")
    print(f"Args: {args}")
    
    # Setup device
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading --- #
    fold_dataloaders_list = None
    test_loader = None
    train_loader = None 
    is_cv = args.k_folds > 1

    curriculum_schedule = None
    if args.use_curriculum:
        if not args.use_synthetic:
             print("Warning: --use-curriculum specified without --use-synthetic. Curriculum schedule ignored.")
        else:
            try:
                curriculum_schedule = parse_curriculum_schedule(args.curriculum_schedule)
                print(f"Parsed curriculum schedule: {curriculum_schedule}")
                if not curriculum_schedule:
                     print("Warning: --use-curriculum specified but schedule is empty or invalid. Using simple augmentation.")
                     args.use_curriculum = False
            except ValueError as e:
                print(f"Error parsing curriculum schedule: {e}. Aborting.")
                return

    aug_type = "curriculum" if args.use_synthetic and args.use_curriculum and curriculum_schedule else ("augmented" if args.use_synthetic else "baseline")
    base_run_prefix = f"{aug_type}_"

    try:
        if is_cv:
            print(f"Loading data for {args.k_folds}-Fold Cross Validation...")
            if args.use_synthetic:
                if args.use_curriculum and curriculum_schedule:
                    print("Using Phased Augmented K-Fold DataLoaders...")
                    initial_ratio = get_current_synthetic_ratio(0, curriculum_schedule)
                    fold_dataloaders_list, test_loader = get_phased_augmented_kfold_dataloaders(
                        data_dir=args.data_dir, synthetic_dir=args.synthetic_dir,
                        k_folds=args.k_folds, batch_size=args.batch_size, num_workers=args.workers,
                        initial_synthetic_ratio=initial_ratio
                    )
                else:
                    print("Using Simple Augmented K-Fold DataLoaders...")
                    fold_dataloaders_list, test_loader = get_simple_augmented_kfold_dataloaders(
                        data_dir=args.data_dir, synthetic_dir=args.synthetic_dir,
                        k_folds=args.k_folds, batch_size=args.batch_size, num_workers=args.workers
                )
            else:
                print("Using Baseline K-Fold DataLoaders...")
                fold_dataloaders_list, test_loader = get_kfold_dataloaders(
                    data_dir=args.data_dir, k_folds=args.k_folds,
                    batch_size=args.batch_size, num_workers=args.workers
                )
        else:
            print("Loading data for single Train/Test split...")
            if args.use_synthetic:
                 if args.use_curriculum:
                      print("Warning: Curriculum learning typically uses K-Fold CV. Running on single split.")
                      print("Falling back to Simple Augmented DataLoaders for non-CV curriculum run...")
                      train_loader, test_loader = get_simple_augmented_dataloaders(
                          data_dir=args.data_dir, synthetic_dir=args.synthetic_dir,
                          batch_size=args.batch_size, num_workers=args.workers
                      )
                 else:
                      print("Using Simple Augmented DataLoaders...")
                      train_loader, test_loader = get_simple_augmented_dataloaders(
                          data_dir=args.data_dir, synthetic_dir=args.synthetic_dir,
                          batch_size=args.batch_size, num_workers=args.workers
                )
            else:
                print("Using Baseline DataLoaders...")
                print("Warning: Using test set as validation for non-CV run. Create a proper validation split.")
                _train_loader, _test_loader = get_dataloaders(args.data_dir, batch_size=args.batch_size, num_workers=args.workers)
                dataloaders = {'train': _train_loader, 'val': _test_loader}
                test_loader = _test_loader

    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Model Training & Evaluation --- #
    all_fold_histories = []
    all_fold_metrics = []

    if is_cv:
        for fold in range(args.k_folds):
            print(f"\n===== Fold {fold + 1} / {args.k_folds} =====")
            model = create_resnet50_baseline(num_classes=2, freeze_base=not args.unfreeze).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            criterion = nn.CrossEntropyLoss()
            current_fold_loaders = {}
            if args.use_curriculum and args.use_synthetic and curriculum_schedule:
                 current_fold_loaders = {
                     'train': fold_dataloaders_list[fold]['train_loader'],
                     'val': fold_dataloaders_list[fold]['val_loader']
                 }
            else:
                 current_fold_loaders = fold_dataloaders_list[fold]

            fold_model, fold_history = train_model(
                model, criterion, optimizer, current_fold_loaders, device,
                num_epochs=args.epochs, scheduler=None,
                model_save_path=args.model_dir, results_save_path=args.results_dir,
                fold=(fold + 1), use_synthetic=args.use_synthetic, curriculum_schedule=curriculum_schedule
            )
            all_fold_histories.append(fold_history)

            print(f"\n--- Evaluating Fold {fold + 1} Model on Test Set ---")
            fold_test_metrics = evaluate_model(fold_model, test_loader, device, criterion)
            all_fold_metrics.append(fold_test_metrics)
            print("-" * 30)

        # --- CV Aggregation & Final Saving --- # 
        # Calculate average and std dev of metrics across folds
        avg_metrics = {key: np.mean([m[key] for m in all_fold_metrics]) for key in all_fold_metrics[0]}
        std_metrics = {key: np.std([m[key] for m in all_fold_metrics]) for key in all_fold_metrics[0]}

        cv_summary = {
            'folds': all_fold_metrics,
            'average': avg_metrics,
            'std_dev': std_metrics
        }

        print("\n===== Cross-Validation Summary =====")
        for key in avg_metrics:
            print(f"Average {key}: {avg_metrics[key]:.4f} +/- {std_metrics[key]:.4f}")

        # Save CV summary
        cv_summary_filename = os.path.join(args.results_dir, f'{base_run_prefix}cv_summary.json')
        with open(cv_summary_filename, 'w') as f:
            json.dump(cv_summary, f, indent=4)
        print(f"Saved CV summary to {cv_summary_filename}")

        # Generate CV plots
        print("\nGenerating CV plots...")
        generate_plots(args.results_dir, args.figures_dir, run_prefix=base_run_prefix, k_folds=args.k_folds)

    else: # Single run (non-CV)
        print("\n===== Starting Single Training Run =====")
        model = create_resnet50_baseline(num_classes=2, freeze_base=not args.unfreeze).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        final_model, history = train_model(
            model, criterion, optimizer, dataloaders, device,
            num_epochs=args.epochs, scheduler=None, # scheduler,
            model_save_path=args.model_dir, results_save_path=args.results_dir,
            fold=None, use_synthetic=args.use_synthetic, curriculum_schedule=curriculum_schedule
        )

        print("\n--- Evaluating Final Model on Test Set ---")
        final_metrics = evaluate_model(final_model, test_loader, device, criterion)

        # Save final metrics
        metrics_data = {
            'config': vars(args),
            'metrics': final_metrics
        }
        final_metrics_filename = os.path.join(args.results_dir, f'{base_run_prefix}final_metrics.json')
        with open(final_metrics_filename, 'w') as f:
            json.dump(metrics_data, f, indent=4)
        print(f"Saved final metrics to {final_metrics_filename}")

        # Generate plots for single run
        print("\nGenerating plots for single run...")
        generate_plots(args.results_dir, args.figures_dir, run_prefix=base_run_prefix, k_folds=None)

    print("\nClassifier training script finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ResNet50 Classifier for Pneumonia Detection')
    
    # Data & Paths
    parser.add_argument('--data-dir', type=str, default='./data/processed',
                        help='Path to the processed dataset directory (default: ./data/processed)')
    parser.add_argument('--synthetic-dir', type=str, default='./data/synthetic',
                        help='Path to the directory containing synthetic images (default: ./data/synthetic)')
    parser.add_argument('--model-dir', type=str, default='./models',
                        help='Directory to save model checkpoints (default: ./models)')
    parser.add_argument('--results-dir', type=str, default='./results/metrics',
                        help='Directory to save training history and metrics (default: ./results/metrics)')
    parser.add_argument('--figures-dir', type=str, default='./results/figures',
                        help='Directory to save generated plots (default: ./results/figures)')
    
    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs (default: 15)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training and evaluation (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for Adam optimizer (default: 0.001)')
    parser.add_argument('--unfreeze', action='store_true', help='Unfreeze base ResNet layers for fine-tuning')
    
    # Cross-Validation
    parser.add_argument('--k-folds', type=int, default=5,
                        help='Number of folds for cross-validation. Set to 1 for single train/test split (default: 5)')
    
    # Data Loading
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers (processes) (default: 4)')

    # Augmentation Strategy
    parser.add_argument('--use-synthetic', action='store_true',
                        help='Use synthetic data augmentation (simple concatenation or curriculum)')
    parser.add_argument('--use-curriculum', action='store_true',
                        help='Use phased curriculum learning for synthetic data (requires --use-synthetic)')
    parser.add_argument('--curriculum-schedule', type=str, default="0:0.0, 5:0.25, 10:0.5",
                        help='Schedule for curriculum learning as \"epoch1:ratio1,epoch2:ratio2,...". Example: \"0:0.0,5:0.25,10:0.5\" (default: \"0:0.0, 5:0.25, 10:0.5\")')

    # Device
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage even if CUDA is available')

    args = parser.parse_args()

    # --- Basic Validation --- #
    if args.k_folds < 1:
        print("Error: k-folds must be at least 1.")
        exit()
    if args.use_curriculum and not args.use_synthetic:
        print("Warning: --use-curriculum requires --use-synthetic. Ignoring curriculum schedule.")
        args.use_curriculum = False

    main(args)
