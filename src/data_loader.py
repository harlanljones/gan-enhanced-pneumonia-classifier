import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from sklearn.model_selection import KFold
import sys
from pathlib import Path
import pandas as pd
from PIL import Image

# Get the absolute path of the project root directory
ROOT_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(ROOT_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Define transformations for ResNet50
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)), # ResNet50 input size
        transforms.RandomHorizontalFlip(), # Basic augmentation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class RSNAPneumoniaDataset(Dataset):
    """Custom Dataset for RSNA Pneumonia data using metadata files."""
    
    def __init__(self, data_dir, metadata_file, transform=None, is_test=False):
        """
        Args:
            data_dir (string): Directory with all the images
            metadata_file (string): Path to the metadata CSV file
            transform (callable, optional): Optional transform to be applied on a sample
            is_test (bool): Whether this is the test dataset (has different metadata format)
        """
        self.data_dir = data_dir
        self.metadata = pd.read_csv(metadata_file)
        self.transform = transform
        self.is_test = is_test
        
        # Print dataset statistics
        print("\nDataset Statistics:")
        print("Total samples:", len(self.metadata))
        
        if not is_test:
            # Training data has class labels
            class_counts = self.metadata['class'].value_counts()
            print("Class distribution:")
            for class_name, count in class_counts.items():
                print(f"- {class_name}: {count}")
            
            # For binary classification:
            # - 'Lung Opacity' -> 1 (Pneumonia)
            # - Everything else -> 0 (Not Pneumonia)
            self.metadata['label'] = (self.metadata['class'] == 'Lung Opacity').astype(int)
        else:
            # Test data has PredictionString column
            # Format: "confidence x y width height" or empty for no finding
            # We'll consider any non-empty PredictionString as a positive case (1)
            self.metadata['label'] = (self.metadata['PredictionString'].str.strip() != '0.5 0 0 100 100').astype(int)
            label_counts = self.metadata['label'].value_counts()
            print("Label distribution:")
            for label, count in label_counts.items():
                print(f"- Class {label}: {count}")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        patient_id = self.metadata.iloc[idx]['patientId']
        img_path = os.path.join(self.data_dir, f"{patient_id}.png")
        
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: Image not found: {img_path}")
            # Return a black image as a fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        label = self.metadata.iloc[idx]['label']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def check_dataset_availability(data_dir=PROCESSED_DIR):
    """
    Checks if the dataset is available in the expected structure.
    
    Args:
        data_dir (str): Directory to check for the dataset
        
    Returns:
        bool: True if the dataset is available, False otherwise
    """
    # Check for required files and directories
    train_metadata = os.path.join(data_dir, 'stage2_train_metadata.csv')
    test_metadata = os.path.join(data_dir, 'stage2_test_metadata.csv')
    train_dir = os.path.join(data_dir, 'Training', 'Images')
    test_dir = os.path.join(data_dir, 'Test')
    
    if not all(os.path.exists(p) for p in [train_metadata, test_metadata, train_dir, test_dir]):
        print(f"Dataset not found in {data_dir} with expected structure.")
        print("Required files/directories:")
        print("- stage2_train_metadata.csv")
        print("- stage2_test_metadata.csv")
        print("- Training/Images/")
        print("- Test/")
        print("\nPlease download and process the RSNA Pneumonia dataset using the download_dataset.py script:")
        print("python src/download_dataset.py")
        return False
    
    # Check if there are images in the directories
    train_images = [f for f in os.listdir(train_dir) if f.endswith('.png')]
    test_images = [f for f in os.listdir(test_dir) if f.endswith('.png')]
    
    if not train_images or not test_images:
        print(f"No images found in Training/Images/ or Test/ directories.")
        print("Please check the dataset structure.")
        return False
    
    print(f"Dataset found with structure:")
    print(f"- Training images: {len(train_images)}")
    print(f"- Test images: {len(test_images)}")
    return True

def get_dataloaders(data_dir=PROCESSED_DIR, batch_size=32, num_workers=4):
    """
    Creates training and testing DataLoaders using metadata files.
    
    Args:
        data_dir (str): Directory containing the processed dataset
        batch_size (int): Batch size for training and testing
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    if not check_dataset_availability(data_dir):
        raise FileNotFoundError(f"Dataset not available in {data_dir}. Please download using the provided script.")
    
    # Create datasets using metadata
    train_dataset = RSNAPneumoniaDataset(
        os.path.join(data_dir, 'Training', 'Images'),
        os.path.join(data_dir, 'stage2_train_metadata.csv'),
        transform=data_transforms['train'],
        is_test=False
    )
    
    test_dataset = RSNAPneumoniaDataset(
        os.path.join(data_dir, 'Test'),
        os.path.join(data_dir, 'stage2_test_metadata.csv'),
        transform=data_transforms['test'],
        is_test=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    return train_loader, test_loader

def get_kfold_dataloaders(data_dir=PROCESSED_DIR, k_folds=5, batch_size=32, num_workers=4):
    """
    Creates DataLoaders for k-fold cross-validation using metadata files.
    
    Args:
        data_dir (str): Directory containing the processed dataset
        k_folds (int): Number of folds for cross-validation
        batch_size (int): Batch size for training and testing
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (fold_dataloaders, test_loader)
               fold_dataloaders: List of dictionaries containing 'train' and 'val' DataLoaders for each fold
               test_loader: DataLoader for the held-out test set
    """
    if not check_dataset_availability(data_dir):
        raise FileNotFoundError(f"Dataset not available in {data_dir}. Please download using the provided script.")
    
    # Create the full training dataset
    full_train_dataset = RSNAPneumoniaDataset(
        os.path.join(data_dir, 'Training', 'Images'),
        os.path.join(data_dir, 'stage2_train_metadata.csv'),
        transform=data_transforms['train'],
        is_test=False
    )
    
    # Create test dataset
    test_dataset = RSNAPneumoniaDataset(
        os.path.join(data_dir, 'Test'),
        os.path.join(data_dir, 'stage2_test_metadata.csv'),
        transform=data_transforms['test'],
        is_test=True
    )
    
    # Create K-fold splits
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_dataloaders = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(full_train_dataset)))):
        print(f"Fold {fold+1}/{k_folds}")
        
        # Create train subset with training transforms
        train_subset = torch.utils.data.Subset(full_train_dataset, train_idx)
        
        # Create validation subset with test transforms
        val_dataset = RSNAPneumoniaDataset(
            os.path.join(data_dir, 'Training', 'Images'),
            os.path.join(data_dir, 'stage2_train_metadata.csv'),
            transform=data_transforms['test'],  # Use test transforms for validation
            is_test=False
        )
        val_subset = torch.utils.data.Subset(val_dataset, val_idx)
        
        # Create data loaders for this fold
        train_loader = DataLoader(
            train_subset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_subset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers, pin_memory=True
        )
        
        fold_dataloaders.append({
            'train': train_loader,
            'val': val_loader
        })
    
    # Create the test loader
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    return fold_dataloaders, test_loader

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test data loader for RSNA Pneumonia dataset')
    parser.add_argument('--data-dir', type=str, default=PROCESSED_DIR,
                        help=f'Path to processed dataset directory (default: {PROCESSED_DIR})')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for testing (default: 4)')
    parser.add_argument('--k-folds', type=int, default=3, help='Number of folds for CV testing (default: 3)')
    
    args = parser.parse_args()
    
    print(f"Project root directory: {ROOT_DIR}")
    print(f"Using data directory: {args.data_dir}")
    
    print("\n--- Checking dataset availability ---")
    dataset_available = check_dataset_availability(args.data_dir)
    
    if dataset_available:
        print("\n--- Testing get_dataloaders ---")
        try:
            train_loader, test_loader = get_dataloaders(args.data_dir, batch_size=args.batch_size)
            print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
            
            # Sample a batch to verify data format
            print("\nSampling a batch from train_loader...")
            dataiter = iter(train_loader)
            images, labels = next(dataiter)
            print(f"Sample batch - Images shape: {images.shape}, Labels: {labels}")
            print(f"Label distribution in sample: {torch.bincount(labels)}")
            
        except Exception as e:
            print(f"Error using get_dataloaders: {e}")
            import traceback
            traceback.print_exc()

        print("\n--- Testing get_kfold_dataloaders ---")
        try:
            fold_loaders, final_test_loader = get_kfold_dataloaders(args.data_dir, k_folds=args.k_folds, batch_size=args.batch_size)
            print(f"Generated {len(fold_loaders)} folds.")
            print(f"Fold 1 - Train batches: {len(fold_loaders[0]['train'])}, Val batches: {len(fold_loaders[0]['val'])}")
            
            # Sample a batch from first fold to verify data format
            print("\nSampling a batch from fold 1 train_loader...")
            dataiter = iter(fold_loaders[0]['train'])
            images, labels = next(dataiter)
            print(f"Sample batch - Images shape: {images.shape}, Labels: {labels}")
            print(f"Label distribution in sample: {torch.bincount(labels)}")
            
            if final_test_loader:
                print(f"Final Test loader batches: {len(final_test_loader)}")
            else:
                print("Final Test loader not created (test directory might be missing).")
        except Exception as e:
            print(f"Error using get_kfold_dataloaders: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Please download the dataset first using the download_dataset.py script.")
