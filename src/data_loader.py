import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset, random_split
import os
import numpy as np
from sklearn.model_selection import KFold
import sys
from pathlib import Path
import pandas as pd
from PIL import Image
import math

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

class SyntheticDataset(Dataset):
    """Dataset for loading generated synthetic images."""
    def __init__(self, synthetic_dir, transform=None, label=1):
        """
        Args:
            synthetic_dir (string): Directory with all the synthetic images.
            transform (callable, optional): Optional transform to be applied on a sample.
            label (int): The label to assign to all synthetic images (default: 1, for 'Lung Opacity').
        """
        self.synthetic_dir = synthetic_dir
        self.image_files = [os.path.join(synthetic_dir, f) for f in os.listdir(synthetic_dir) if f.endswith('.png')]
        self.transform = transform
        self.label = label
        print(f"Found {len(self.image_files)} synthetic images in {synthetic_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Error loading synthetic image {img_path}: {e}")
            # Return a black image as a fallback
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        return image, self.label

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

# --- Functions for Augmented Data ---

def get_augmented_dataloaders(data_dir=PROCESSED_DIR, synthetic_dir='data/synthetic', batch_size=32, num_workers=4):
    """
    Creates training and testing DataLoaders using metadata files, augmenting the training set
    with synthetic images. The test set remains unchanged.

    Args:
        data_dir (str): Directory containing the original processed dataset.
        synthetic_dir (str): Directory containing the generated synthetic images.
        batch_size (int): Batch size for training and testing.
        num_workers (int): Number of workers for data loading.

    Returns:
        tuple: (train_loader, test_loader)
    """
    if not check_dataset_availability(data_dir):
        raise FileNotFoundError(f"Original dataset not available in {data_dir}.")
    if not os.path.exists(synthetic_dir) or not os.listdir(synthetic_dir):
         raise FileNotFoundError(f"Synthetic dataset directory {synthetic_dir} is empty or does not exist. Generate images first.")


    # Create original training dataset
    original_train_dataset = RSNAPneumoniaDataset(
        os.path.join(data_dir, 'Training', 'Images'),
        os.path.join(data_dir, 'stage2_train_metadata.csv'),
        transform=data_transforms['train'],
        is_test=False
    )

    # Create synthetic dataset (assuming label 1 for pneumonia)
    synthetic_dataset = SyntheticDataset(
        synthetic_dir,
        transform=data_transforms['train'] # Use same transforms as training
    )

    # Combine original training and synthetic datasets
    augmented_train_dataset = torch.utils.data.ConcatDataset([original_train_dataset, synthetic_dataset])

    # Create original test dataset (unchanged)
    test_dataset = RSNAPneumoniaDataset(
        os.path.join(data_dir, 'Test'),
        os.path.join(data_dir, 'stage2_test_metadata.csv'),
        transform=data_transforms['test'],
        is_test=True
    )

    # Create data loaders
    train_loader = DataLoader(
        augmented_train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )

    print(f"Original train dataset size: {len(original_train_dataset)}")
    print(f"Synthetic dataset size: {len(synthetic_dataset)}")
    print(f"Augmented train dataset size: {len(augmented_train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return train_loader, test_loader


def get_augmented_kfold_dataloaders(data_dir=PROCESSED_DIR, synthetic_dir='data/synthetic', k_folds=5, batch_size=32, num_workers=4):
    """
    Creates DataLoaders for k-fold cross-validation, augmenting each training fold
    with synthetic images. Validation and test sets contain only original data.

    Args:
        data_dir (str): Directory containing the original processed dataset.
        synthetic_dir (str): Directory containing the generated synthetic images.
        k_folds (int): Number of folds for cross-validation.
        batch_size (int): Batch size for training and testing.
        num_workers (int): Number of workers for data loading.

    Returns:
        tuple: (fold_dataloaders, test_loader)
               fold_dataloaders: List of dictionaries containing 'train' (augmented) and 'val' (original) DataLoaders for each fold.
               test_loader: DataLoader for the held-out test set (original).
    """
    if not check_dataset_availability(data_dir):
        raise FileNotFoundError(f"Original dataset not available in {data_dir}.")
    if not os.path.exists(synthetic_dir) or not os.listdir(synthetic_dir):
         raise FileNotFoundError(f"Synthetic dataset directory {synthetic_dir} is empty or does not exist. Generate images first.")

    # Create the full original training dataset (used for splitting)
    full_train_dataset = RSNAPneumoniaDataset(
        os.path.join(data_dir, 'Training', 'Images'),
        os.path.join(data_dir, 'stage2_train_metadata.csv'),
        transform=data_transforms['train'], # Base transforms for subset creation
        is_test=False
    )

    # Create synthetic dataset (to be added to each training fold)
    synthetic_dataset = SyntheticDataset(
        synthetic_dir,
        transform=data_transforms['train'] # Use same transforms as training
    )
    print(f"Synthetic dataset size: {len(synthetic_dataset)}")


    # Create original test dataset (unchanged)
    test_dataset = RSNAPneumoniaDataset(
        os.path.join(data_dir, 'Test'),
        os.path.join(data_dir, 'stage2_test_metadata.csv'),
        transform=data_transforms['test'],
        is_test=True
    )

    # Create K-fold splits on the original training data
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_dataloaders = []

    original_indices = list(range(len(full_train_dataset)))

    for fold, (train_idx, val_idx) in enumerate(kf.split(original_indices)):
        print(f"\nFold {fold+1}/{k_folds}")

        # --- Create Training Subset (Original) ---
        # Need to wrap Subset with a dataset applying TRAIN transforms
        train_subset_original = torch.utils.data.Subset(full_train_dataset, train_idx)

        # --- Augment Training Subset ---
        # Combine the original training subset with ALL synthetic images
        augmented_train_fold_dataset = torch.utils.data.ConcatDataset([train_subset_original, synthetic_dataset])
        print(f"  Augmented Train Fold Size: {len(augmented_train_fold_dataset)} (Original: {len(train_subset_original)}, Synthetic: {len(synthetic_dataset)})")


        # --- Create Validation Subset (Original) ---
        # Create a separate dataset instance for validation subset with TEST transforms
        val_dataset_instance = RSNAPneumoniaDataset(
            os.path.join(data_dir, 'Training', 'Images'),
            os.path.join(data_dir, 'stage2_train_metadata.csv'),
            transform=data_transforms['test'], # Use test transforms for validation
            is_test=False
        )
        val_subset = torch.utils.data.Subset(val_dataset_instance, val_idx)
        print(f"  Validation Fold Size (Original): {len(val_subset)}")


        # Create data loaders for this fold
        train_loader = DataLoader(
            augmented_train_fold_dataset, batch_size=batch_size,
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

    # Create the final test loader (original data)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    print(f"\nTest dataset size (Original): {len(test_dataset)}")


    return fold_dataloaders, test_loader

class PhasedAugmentedDataset(Dataset):
    """
    Dataset that combines a real dataset and a synthetic dataset,
    allowing dynamic adjustment of the synthetic data ratio.
    Assumes synthetic data always belongs to the positive class (label=1).
    """
    def __init__(self, real_dataset: Dataset, synthetic_dataset: Dataset, synthetic_ratio: float = 0.0):
        """
        Args:
            real_dataset (Dataset): The dataset containing real images.
            synthetic_dataset (Dataset): The dataset containing synthetic images (assumed positive class).
            synthetic_ratio (float): Initial proportion of the batch that should be synthetic (0.0 to 1.0).
        """
        self.real_dataset = real_dataset
        self.synthetic_dataset = synthetic_dataset
        self._set_synthetic_ratio(synthetic_ratio)

        # Pre-calculate indices for faster sampling
        self.real_positive_indices = []
        self.real_negative_indices = []
        try:
            print("Calculating positive/negative indices for real dataset...")
            
            # Check if we're dealing with RSNAPneumoniaDataset or its Subset
            if isinstance(real_dataset, Subset) and isinstance(real_dataset.dataset, RSNAPneumoniaDataset):
                # For Subset of RSNAPneumoniaDataset, use metadata but only for subset indices
                base_dataset = real_dataset.dataset
                subset_indices = real_dataset.indices
                for idx in subset_indices:
                    label = base_dataset.metadata.iloc[idx]['label']
                    if label == 1:
                        self.real_positive_indices.append(idx)
                    else:
                        self.real_negative_indices.append(idx)
            elif isinstance(real_dataset, RSNAPneumoniaDataset):
                # For direct RSNAPneumoniaDataset, use metadata directly
                for idx, row in real_dataset.metadata.iterrows():
                    if row['label'] == 1:
                        self.real_positive_indices.append(idx)
                    else:
                        self.real_negative_indices.append(idx)
            else:
                # Fallback for other dataset types: iterate through dataset
                print("Warning: Dataset type not optimized, falling back to iteration...")
                for i in range(len(real_dataset)):
                    item = real_dataset[i]
                    if isinstance(item, tuple) and len(item) == 2:
                        label = item[1]
                    else:
                        label = item
                    
                    if label == 1:
                        self.real_positive_indices.append(i)
                    else:
                        self.real_negative_indices.append(i)
            
            print(f"Real dataset breakdown: {len(self.real_positive_indices)} positive, {len(self.real_negative_indices)} negative samples.")
            if not self.real_positive_indices:
                print("Warning: No positive samples found in the real dataset.")
            if not self.real_negative_indices:
                print("Warning: No negative samples found in the real dataset.")

        except Exception as e:
            print(f"Warning: Could not pre-calculate positive/negative indices for real dataset: {e}. Sampling randomly.")
            self.real_positive_indices = list(range(len(real_dataset)))  # Fallback
            self.real_negative_indices = []  # Cannot guarantee negative samples

        if not self.synthetic_dataset:
            print("Warning: Synthetic dataset is empty or None.")

    def _set_synthetic_ratio(self, synthetic_ratio: float):
        self.synthetic_ratio = max(0.0, min(1.0, synthetic_ratio))
        print(f"PhasedAugmentedDataset: Set synthetic ratio to {self.synthetic_ratio:.2f}")

    def set_synthetic_ratio(self, synthetic_ratio: float):
        """Public method to update the synthetic ratio."""
        self._set_synthetic_ratio(synthetic_ratio)

    def __len__(self):
        # Effective length balances real and synthetic based on the ratio.
        # A common strategy is to keep the total number of positive samples roughly constant
        # or scale based on the real dataset size. Here, let's aim to keep epoch size
        # similar to the real dataset, adjusting the mix within batches.
        # The DataLoader will handle batch creation. The dataset length itself
        # can represent the pool we draw from. Let's use the real dataset length.
        return len(self.real_dataset)


    def __getitem__(self, idx):
        # This idx is mainly used by the DataLoader's sampler.
        # We override the sampling logic here based on the ratio.

        # Determine if the sample should be synthetic based on the ratio
        if np.random.rand() < self.synthetic_ratio:
            # Fetch a synthetic sample (always positive class)
            if len(self.synthetic_dataset) > 0:
                synth_idx = np.random.randint(len(self.synthetic_dataset))
                return self.synthetic_dataset[synth_idx]
            else:
                # Fallback if synthetic dataset is empty: fetch a real positive sample
                 if self.real_positive_indices:
                     real_idx = np.random.choice(self.real_positive_indices)
                     return self.real_dataset[real_idx]
                 else:
                     # Fallback: fetch any real sample if no positives known
                     real_idx = np.random.randint(len(self.real_dataset))
                     return self.real_dataset[real_idx]
        else:
            # Fetch a real sample (maintain original class distribution for the real part)
            # Use the provided idx (modulo length) to sample from real dataset,
            # respecting the sampler's intention if possible (e.g., for shuffling).
            real_idx = idx % len(self.real_dataset)
            return self.real_dataset[real_idx]

def get_simple_augmented_dataloaders(data_dir=PROCESSED_DIR, synthetic_dir=os.path.join(DATA_DIR, 'synthetic'),
                                     batch_size=32, num_workers=4):
    """
    Creates training and testing DataLoaders using metadata files,
    simply concatenating all synthetic data to the training set.
    (Original augmentation strategy)
    """
    if not check_dataset_availability(data_dir):
        raise FileNotFoundError(f"Dataset not available in {data_dir}.")

    # Create real datasets
    train_dataset_real = RSNAPneumoniaDataset(
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

    # Create synthetic dataset
    synthetic_dataset = SyntheticDataset(
        synthetic_dir,
        transform=data_transforms['train'] # Apply training transforms
    )

    if len(synthetic_dataset) == 0:
        print("Warning: No synthetic images found. Training with real data only.")
        augmented_train_dataset = train_dataset_real
    else:
        # Combine real training data and synthetic data
        augmented_train_dataset = ConcatDataset([train_dataset_real, synthetic_dataset])

    # Create data loaders
    train_loader = DataLoader(
        augmented_train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )

    print(f"Augmented Train dataset size: {len(augmented_train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return train_loader, test_loader


def get_simple_augmented_kfold_dataloaders(data_dir=PROCESSED_DIR, synthetic_dir=os.path.join(DATA_DIR, 'synthetic'),
                                           k_folds=5, batch_size=32, num_workers=4):
    """
    Creates DataLoaders for k-fold cross-validation, simply concatenating
    all synthetic data to the training fold.
    (Original augmentation strategy for CV)
    """
    if not check_dataset_availability(data_dir):
        raise FileNotFoundError(f"Dataset not available in {data_dir}.")

    # Create the full real training dataset
    full_train_dataset_real = RSNAPneumoniaDataset(
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

    # Create synthetic dataset
    synthetic_dataset = SyntheticDataset(
        synthetic_dir,
        transform=data_transforms['train']
    )
    if len(synthetic_dataset) == 0:
        print("Warning: No synthetic images found. Proceeding with real data only for CV.")

    # Create K-fold splits on the *real* data
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    indices = list(range(len(full_train_dataset_real)))
    fold_dataloaders = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\n--- Fold {fold+1}/{k_folds} ---")

        # Create subsets for real train and validation
        real_train_subset = Subset(full_train_dataset_real, train_idx)
        real_val_subset = Subset(full_train_dataset_real, val_idx)
        # Important: Apply test transform to validation set
        real_val_subset.dataset.transform = data_transforms['test']

        # Combine real training subset with *all* synthetic data for this fold
        if len(synthetic_dataset) > 0:
            fold_train_dataset = ConcatDataset([real_train_subset, synthetic_dataset])
        else:
            fold_train_dataset = real_train_subset # No synthetic data

        fold_train_loader = DataLoader(
            fold_train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True
        )
        fold_val_loader = DataLoader(
            real_val_subset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers, pin_memory=True
        )

        fold_dataloaders.append({'train': fold_train_loader, 'val': fold_val_loader})
        print(f"Fold {fold+1} - Train size: {len(fold_train_dataset)}, Val size: {len(real_val_subset)}")
        # Reset transform for next fold if validation set modified it
        real_val_subset.dataset.transform = data_transforms['train']


    # Test loader remains the same
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )

    print(f"\nTest dataset size: {len(test_dataset)}")

    return fold_dataloaders, test_loader


def get_phased_augmented_kfold_dataloaders(data_dir=PROCESSED_DIR, synthetic_dir=os.path.join(DATA_DIR, 'synthetic'),
                                           k_folds=5, batch_size=32, num_workers=4, initial_synthetic_ratio=0.0):
    """
    Creates DataLoaders for k-fold cross-validation using PhasedAugmentedDataset.
    Allows dynamic adjustment of synthetic ratio during training.
    """
    if not check_dataset_availability(data_dir):
        raise FileNotFoundError(f"Dataset not available in {data_dir}.")

    # Create the full real training dataset
    full_train_dataset_real = RSNAPneumoniaDataset(
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

    # Create synthetic dataset
    synthetic_dataset = SyntheticDataset(
        synthetic_dir,
        transform=data_transforms['train']
    )
    if len(synthetic_dataset) == 0:
        print("Warning: No synthetic images found. Curriculum learning will use real data only.")

    # Create K-fold splits on the *real* data
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    indices = list(range(len(full_train_dataset_real)))
    fold_dataloaders = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\n--- Fold {fold+1}/{k_folds} ---")

        # Create subsets for real train and validation
        real_train_subset = Subset(full_train_dataset_real, train_idx)
        real_val_subset = Subset(full_train_dataset_real, val_idx)
        # Important: Apply test transform to validation set
        real_val_subset.dataset.transform = data_transforms['test'] # Apply test transform

        # Create PhasedAugmentedDataset for this fold's training
        fold_train_phased_dataset = PhasedAugmentedDataset(
            real_dataset=real_train_subset,
            synthetic_dataset=synthetic_dataset,
            synthetic_ratio=initial_synthetic_ratio
        )

        fold_train_loader = DataLoader(
            fold_train_phased_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True
        )
        fold_val_loader = DataLoader(
            real_val_subset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers, pin_memory=True
        )

        # Store the dataset itself to allow ratio updates later
        fold_dataloaders.append({
            'train_loader': fold_train_loader,
            'val_loader': fold_val_loader,
            'train_dataset': fold_train_phased_dataset # Reference to the dataset
        })
        print(f"Fold {fold+1} - Real Train size: {len(real_train_subset)}, Val size: {len(real_val_subset)}")
        print(f"Fold {fold+1} - Initial synthetic ratio: {initial_synthetic_ratio:.2f}")
        # Reset transform for next fold if validation set modified it
        real_val_subset.dataset.transform = data_transforms['train']


    # Test loader remains the same
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )

    print(f"\nTest dataset size: {len(test_dataset)}")

    return fold_dataloaders, test_loader

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test data loader for RSNA Pneumonia dataset')
    parser.add_argument('--data-dir', type=str, default=PROCESSED_DIR,
                        help=f'Path to processed dataset directory (default: {PROCESSED_DIR})')
    parser.add_argument('--synthetic-dir', type=str, default=os.path.join(DATA_DIR, 'synthetic'), help='Path to synthetic dataset directory')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for testing (default: 4)')
    parser.add_argument('--k-folds', type=int, default=3, help='Number of folds for CV testing (default: 3)')
    parser.add_argument('--test-mode', type=str, choices=['basic', 'kfold', 'augmented', 'kfold_augmented', 'phased_kfold'], default='basic', help='Which dataloader function to test')
    
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
