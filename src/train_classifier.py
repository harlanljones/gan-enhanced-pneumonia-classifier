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
from data_loader import get_dataloaders, get_kfold_dataloaders
from classifier import create_resnet50_baseline

def train_model(model, criterion, optimizer, dataloaders, device, num_epochs=25, scheduler=None, model_save_path='../models', results_save_path='../results/metrics', fold=None):
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
                best_model_filename = os.path.join(model_save_path, f'{fold_prefix}best_baseline_resnet50.pth')
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
    history_filename = os.path.join(results_save_path, f'{fold_prefix}training_history.json')
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

def load_history(filepath):
    """Loads training history from a JSON file."""
    try:
        with open(filepath, 'r') as f:
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
        print(f"Warning: History file not found: {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from: {filepath}")
        return None
    except Exception as e:
        print(f"Warning: An error occurred loading {filepath}: {e}")
        return None

def plot_metric(histories, metric_key_base, title, ylabel, output_path):
    """Plots a specific metric (loss or accuracy) for multiple training histories."""
    plt.figure(figsize=(10, 6))
    num_epochs = 0
    plot_has_data = False # Flag to check if any data was actually plotted

    for i, history in enumerate(histories):
        if history:
            train_key = f'train_{metric_key_base}'
            val_key = f'val_{metric_key_base}'
            fold_label = f"Fold {i+1}" if len(histories) > 1 else "Run"

            if train_key in history and val_key in history and len(history[train_key]) > 0:
                epochs = range(1, len(history[train_key]) + 1)
                num_epochs = max(num_epochs, len(epochs)) # Track max epochs across folds
                plt.plot(epochs, history[train_key], label=f'{fold_label} Train {metric_key_base.capitalize()}')
                plt.plot(epochs, history[val_key], linestyle='--', label=f'{fold_label} Val {metric_key_base.capitalize()}')
                plot_has_data = True
            # else: # Optional: be less verbose if keys/data missing, already warned by load_history
            #     print(f"Warning: Missing keys or empty data for '{train_key}' or '{val_key}' in history {i+1}.")

    if plot_has_data: # Only finalize plot if data was plotted
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.xlim(1, num_epochs) # Adjust x-axis limits based on the actual number of epochs plotted
        if 'acc' in metric_key_base:
             plt.ylim(0, 1.05) # Extend slightly beyond 1.0
        plt.legend(loc='best', fontsize='small') # Adjust legend location and size
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close() # Close the figure to free memory
        print(f"Saved plot to {output_path}")
    else:
        plt.close() # Ensure figure is closed even if no data plotted
        print(f"Skipping plot '{title}' as no valid data was found.")


def plot_cv_summary(cv_summary_path, output_dir):
    """Plots the final test metrics per fold from the CV summary."""
    try:
        with open(cv_summary_path, 'r') as f:
            summary = json.load(f)
    except FileNotFoundError:
        print(f"Warning: CV summary file not found: {cv_summary_path}")
        return
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from: {cv_summary_path}")
        return
    except Exception as e:
        print(f"Warning: An error occurred loading {cv_summary_path}: {e}")
        return

    metrics_per_fold = summary.get('test_metrics_per_fold', [])
    if not metrics_per_fold:
        print("Warning: No 'test_metrics_per_fold' found in CV summary.")
        return

    # Dynamically get metric keys from the first fold (ensure consistency or handle errors)
    if not isinstance(metrics_per_fold[0], dict):
         print("Warning: 'test_metrics_per_fold' does not contain dictionaries.")
         return
    metric_keys = [k for k in metrics_per_fold[0].keys() if k != 'loss'] # Exclude loss for bar plot
    if not metric_keys:
        print("Warning: No suitable metric keys found (excluding 'loss') in CV summary.")
        return

    num_folds = len(metrics_per_fold)
    folds = [f"Fold {i+1}" for i in range(num_folds)]

    num_metrics = len(metric_keys)
    bar_width = 0.8 / num_metrics # Adjust bar width based on number of metrics
    index = np.arange(num_folds)

    plt.figure(figsize=(12, 7))
    plot_has_data = False # Flag to check if any bars were added

    for i, key in enumerate(metric_keys):
        values = [fold_metrics.get(key, np.nan) for fold_metrics in metrics_per_fold]
        # Check if we have valid values before plotting
        if not all(np.isnan(values)):
            plot_key = key.replace('_', ' ').capitalize()
            plt.bar(index + i * bar_width, values, bar_width, label=plot_key)
            plot_has_data = True

    if plot_has_data:
        plt.xlabel('Fold')
        plt.ylabel('Score')
        plt.title('Test Set Metrics per Fold (Cross-Validation)')
        plt.xticks(index + bar_width * (num_metrics - 1) / 2, folds) # Center ticks between bars
        plt.ylim(0, 1.05) # Set y-axis limit for typical metrics (0-1)
        plt.legend(loc='best', fontsize='small')
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()

        output_path = os.path.join(output_dir, 'cv_test_metrics_per_fold.png')
        plt.savefig(output_path)
        plt.close() # Close the figure to free memory
        print(f"Saved plot to {output_path}")
    else:
        plt.close() # Ensure figure is closed even if no data plotted
        print("Skipping CV summary plot as no valid metric data was found.")


def generate_plots(metrics_dir, figures_dir):
    """Loads histories and generates all plots."""
    print(f"\n--- Generating Plots ---")
    print(f"Reading metrics from: {metrics_dir}")
    print(f"Saving figures to: {figures_dir}")
    os.makedirs(figures_dir, exist_ok=True)

    # Find history files
    fold_histories = []
    single_history = None
    cv_summary_path = None

    try:
        if not os.path.isdir(metrics_dir):
            print(f"Error: Metrics directory not found: {metrics_dir}")
            return

        for filename in sorted(os.listdir(metrics_dir)):
            filepath = os.path.join(metrics_dir, filename)
            if filename.startswith('fold_') and filename.endswith('_training_history.json'):
                history = load_history(filepath)
                if history:
                    fold_histories.append(history)
            elif filename == 'training_history.json': # Single run history
                single_history = load_history(filepath)
            elif filename == 'cv_summary_metrics.json':
                cv_summary_path = filepath
    except Exception as e:
        print(f"Error reading metrics directory {metrics_dir}: {e}")
        return


    if fold_histories:
        print(f"Found {len(fold_histories)} fold history files. Plotting combined CV results.")
        plot_metric(fold_histories, 'loss', 'Cross-Validation Training & Validation Loss', 'Loss',
                    os.path.join(figures_dir, 'cv_loss_curves.png'))
        plot_metric(fold_histories, 'acc', 'Cross-Validation Training & Validation Accuracy', 'Accuracy',
                    os.path.join(figures_dir, 'cv_accuracy_curves.png'))
        if cv_summary_path:
            plot_cv_summary(cv_summary_path, figures_dir)
        else:
            print("Info: cv_summary_metrics.json not found. Skipping CV summary plot.")

    elif single_history:
        print("Found single run history file. Plotting results.")
        plot_metric([single_history], 'loss', 'Training & Validation Loss', 'Loss',
                    os.path.join(figures_dir, 'loss_curve.png'))
        plot_metric([single_history], 'acc', 'Training & Validation Accuracy', 'Accuracy',
                    os.path.join(figures_dir, 'accuracy_curve.png'))
        # Optionally plot final evaluation if the file exists
        final_metrics_path = os.path.join(metrics_dir, 'final_evaluation_metrics.json')
        # (Add logic here to plot final_evaluation_metrics if desired, e.g., a single bar chart)

    else:
        print("No training history files (.json) found to generate plots.")

    print(f"--- Plot Generation Complete ---")


# --- End of Plotting Functions ---


def main(args):
    """Main function to orchestrate training, evaluation, and plotting."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    # Ensure results and model directories exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss()

    if args.k_folds > 0: # Changed condition to > 0
        print(f"Starting {args.k_folds}-Fold Cross Validation...")
        fold_dataloaders, test_loader = get_kfold_dataloaders(args.data_dir, k_folds=args.k_folds, batch_size=args.batch_size, num_workers=args.workers)

        if test_loader is None:
            print("ERROR: Test loader could not be created. Check test data directory and structure.")
            return

        all_fold_histories = []
        test_metrics_per_fold = []
        best_fold_val_acc = -1
        best_model_state_overall = None

        for i, fold_data in enumerate(fold_dataloaders):
            print(f"\n===== Fold {i+1}/{args.k_folds} =====")
            # Create a new model instance for each fold
            model = create_resnet50_baseline(num_classes=2, pretrained=True, freeze_base=not args.unfreeze)
            model = model.to(device)

            # Setup optimizer - ensure only trainable parameters are optimized
            params_to_optimize = model.parameters() if not (not args.unfreeze) else model.fc.parameters()
            optimizer = optim.Adam(params_to_optimize, lr=args.lr)
            # Optional: Add a learning rate scheduler
            # scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
            scheduler = None

            fold_model, fold_history = train_model(
                model, criterion, optimizer, fold_data, device, args.epochs, scheduler,
                args.model_dir, args.results_dir, fold=i+1
            )
            all_fold_histories.append(fold_history)

            # Evaluate this fold's best model on the held-out test set
            print(f"\n--- Evaluating Fold {i+1} Best Model on Test Set ---")
            fold_test_metrics = evaluate_model(fold_model, test_loader, device, criterion)
            test_metrics_per_fold.append(fold_test_metrics)

            # Track the overall best model based on validation accuracy
            current_best_val_acc = max(fold_history['val_acc'])
            if current_best_val_acc > best_fold_val_acc:
                best_fold_val_acc = current_best_val_acc
                best_model_state_overall = fold_model.state_dict()
                # Save the overall best model checkpoint
                overall_best_path = os.path.join(args.model_dir, 'best_overall_baseline_resnet50.pth')
                torch.save(best_model_state_overall, overall_best_path)
                print(f"Saved overall best model (val_acc={best_fold_val_acc:.4f}) from fold {i+1} to {overall_best_path}")


        # Aggregate and report CV results
        print("\n===== Cross-Validation Summary ====")
        # Aggregate weighted metrics
        avg_test_metrics = {
            key: np.mean([m[key] for m in test_metrics_per_fold])
            for key in ['accuracy', 'weighted_precision', 'weighted_recall', 'weighted_f1_score']
        }
        std_test_metrics = {
            key: np.std([m[key] for m in test_metrics_per_fold])
             for key in ['accuracy', 'weighted_precision', 'weighted_recall', 'weighted_f1_score']
        }
        # Separately handle loss if needed (was excluded before)
        avg_test_metrics['loss'] = np.mean([m['loss'] for m in test_metrics_per_fold])
        std_test_metrics['loss'] = np.std([m['loss'] for m in test_metrics_per_fold])


        print("Average Test Metrics across folds:")
        print(f"  Loss: {avg_test_metrics['loss']:.4f} +/- {std_test_metrics['loss']:.4f}")
        print(f"  Accuracy: {avg_test_metrics['accuracy']:.4f} +/- {std_test_metrics['accuracy']:.4f}")
        print(f"  Weighted Precision: {avg_test_metrics['weighted_precision']:.4f} +/- {std_test_metrics['weighted_precision']:.4f}")
        print(f"  Weighted Recall: {avg_test_metrics['weighted_recall']:.4f} +/- {std_test_metrics['weighted_recall']:.4f}")
        print(f"  Weighted F1 Score: {avg_test_metrics['weighted_f1_score']:.4f} +/- {std_test_metrics['weighted_f1_score']:.4f}")

        # Save aggregated results
        cv_results_path = os.path.join(args.results_dir, 'cv_summary_metrics.json')
        summary_data = {
            'average_test_metrics': avg_test_metrics,
            'std_dev_test_metrics': std_test_metrics,
            'test_metrics_per_fold': test_metrics_per_fold
        }
        with open(cv_results_path, 'w') as f:
            json.dump(summary_data, f, indent=4)
        print(f"Saved CV summary metrics to {cv_results_path}")

    else:
        # Simple Train/Test Split (No Cross-Validation)
        print("Starting Single Train/Test Split Training...")
        train_loader, test_loader = get_dataloaders(args.data_dir, batch_size=args.batch_size, num_workers=args.workers)

        if test_loader is None:
             print("ERROR: Test loader could not be created. Check test data directory and structure.")
             return

        model = create_resnet50_baseline(num_classes=2, pretrained=True, freeze_base=not args.unfreeze)
        model = model.to(device)

        params_to_optimize = model.parameters() if not (not args.unfreeze) else model.fc.parameters()
        optimizer = optim.Adam(params_to_optimize, lr=args.lr)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        scheduler = None # Keep it simple for now

        dataloaders = {'train': train_loader, 'val': test_loader} # Use test set as validation set here

        best_model, history = train_model(
            model, criterion, optimizer, dataloaders, device, args.epochs, scheduler,
            args.model_dir, args.results_dir, fold=None # No fold number
        )

        # Final evaluation on the test set using the best model from training
        print("\n--- Final Evaluation on Test Set ---")
        final_metrics = evaluate_model(best_model, test_loader, device, criterion)

        # Save final metrics
        final_metrics_path = os.path.join(args.results_dir, 'final_evaluation_metrics.json')
        with open(final_metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=4)
        print(f"Saved final evaluation metrics to {final_metrics_path}")

    # --- Call Plotting Function ---
    try:
        generate_plots(args.results_dir, args.figures_dir)
    except ImportError:
         print("\nWarning: Matplotlib not found. Skipping plot generation.")
         print("Install matplotlib to generate plots: pip install matplotlib")
    except Exception as e:
        print(f"\nAn error occurred during plot generation: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ResNet50 Baseline Classifier and Generate Plots')
    parser.add_argument('--data-dir', type=str, default='./data/processed',
                        help='Path to the processed dataset directory (containing train/test subdirs)')
    parser.add_argument('--model-dir', type=str, default='./models',
                        help='Directory to save model checkpoints')
    parser.add_argument('--results-dir', type=str, default='./results/metrics',
                        help='Directory to save training history and evaluation metrics')
    parser.add_argument('--figures-dir', type=str, default='./results/figures',
                        help='Directory to save generated plot images')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Input batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--unfreeze', action='store_true', help='Unfreeze base ResNet layers for fine-tuning')
    parser.add_argument('--k-folds', type=int, default=5, help='Number of folds for cross-validation (set to 0 to disable)')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--cpu', action='store_true', help='Force use CPU even if CUDA is available')

    args = parser.parse_args()

    # Ensure k_folds=0 means no CV (adjusting logic slightly for clarity)
    if args.k_folds <= 1:
        args.k_folds = 0

    # Check if matplotlib is available before starting potentially long training
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: Matplotlib not found. Plots will not be generated after training.")
        print("Install matplotlib to generate plots: pip install matplotlib")

    main(args)
