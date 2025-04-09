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
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)

    metrics = {
        'loss': eval_loss,
        'accuracy': eval_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    print(f"Evaluation Results - Loss: {eval_loss:.4f}, Accuracy: {eval_accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return metrics

def main(args):
    """Main function to orchestrate training and evaluation."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    criterion = nn.CrossEntropyLoss()

    if args.k_folds > 1:
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
        avg_test_metrics = {key: np.mean([m[key] for m in test_metrics_per_fold]) for key in test_metrics_per_fold[0] if key != 'loss'}
        std_test_metrics = {key: np.std([m[key] for m in test_metrics_per_fold]) for key in test_metrics_per_fold[0] if key != 'loss'}

        print("Average Test Metrics across folds:")
        for key, value in avg_test_metrics.items():
            print(f"  {key.capitalize()}: {value:.4f} +/- {std_test_metrics[key]:.4f}")

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ResNet50 Baseline Classifier')
    parser.add_argument('--data-dir', type=str, default='./data/processed',
                        help='Path to the processed dataset directory (containing train/test subdirs)')
    parser.add_argument('--model-dir', type=str, default='./models',
                        help='Directory to save model checkpoints')
    parser.add_argument('--results-dir', type=str, default='./results/metrics',
                        help='Directory to save training history and evaluation metrics')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Input batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--unfreeze', action='store_true', help='Unfreeze base ResNet layers for fine-tuning')
    parser.add_argument('--k-folds', type=int, default=5, help='Number of folds for cross-validation (set to 1 or 0 to disable)')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--cpu', action='store_true', help='Force use CPU even if CUDA is available')

    args = parser.parse_args()

    # Ensure k_folds=1 means no CV
    if args.k_folds <= 1:
        args.k_folds = 0

    main(args)
