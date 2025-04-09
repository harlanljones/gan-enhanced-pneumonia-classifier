import os
import kaggle
import argparse
import zipfile

def download_kaggle_dataset(dataset_name, output_dir, unzip=True):
    """
    Downloads a dataset from Kaggle using the Kaggle API.
    
    Args:
        dataset_name (str): Name of the dataset on Kaggle (e.g., 'iamtapendu/rsna-pneumonia-processed-dataset')
        output_dir (str): Directory to save the downloaded dataset
        unzip (bool): Whether to unzip the downloaded dataset (default: True)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading dataset '{dataset_name}' to {output_dir}...")
    
    # Download the dataset
    try:
        kaggle.api.authenticate()
        # Ensure unzip is True for this dataset as it contains the structure
        kaggle.api.dataset_download_files(dataset_name, path=output_dir, unzip=True) 
        print(f"Dataset downloaded and extracted successfully to {output_dir}")
    except Exception as e:
        print(f"Error during Kaggle API download: {e}")
        print("Please ensure your Kaggle API token is correctly set up in ~/.kaggle/kaggle.json")
        print("See README for setup instructions.")
        raise # Re-raise the exception to stop the script
    
    # The unzip=True in dataset_download_files handles extraction.
    # Manual unzipping logic is kept just in case unzip=False is ever used,
    # but for this pre-processed dataset, it shouldn't be needed.
    if not unzip:
        print("Manual unzipping requested (unzip=False)...")
        zip_files = [f for f in os.listdir(output_dir) if f.endswith('.zip')]
        if not zip_files:
            print(f"Warning: No zip files found in {output_dir} to unzip manually.")
            return
        
        for zip_file in zip_files:
            zip_path = os.path.join(output_dir, zip_file)
            print(f"Unzipping {zip_path}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)
                print(f"Unzipped {zip_path}")
                # Optionally remove the zip file after extraction
                # os.remove(zip_path)
            except zipfile.BadZipFile:
                print(f"Error: {zip_path} is not a valid zip file or is corrupted.")
            except Exception as e:
                print(f"Error unzipping {zip_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Download RSNA Pneumonia processed dataset from Kaggle')
    parser.add_argument('--dataset', type=str, default='iamtapendu/rsna-pneumonia-processed-dataset',
                        help='Kaggle dataset name (default: iamtapendu/rsna-pneumonia-processed-dataset)')
    # Changed help text for data-dir to reflect it now holds the final structure
    parser.add_argument('--data-dir', type=str, default='./data/processed',
                        help='Directory to save the downloaded and extracted dataset (default: ./data/processed)')
    # Removed raw-dir, processed-dir and no-process arguments as they are no longer relevant
    
    args = parser.parse_args()
    
    # Make sure path is absolute
    data_dir = os.path.abspath(args.data_dir)
    
    try:
        # Download dataset (unzipping happens by default via kaggle api call)
        download_kaggle_dataset(args.dataset, data_dir)
        
        print("\nDataset download completed successfully.")
        print(f"Dataset saved to: {data_dir}")
        print("\nPlease verify the structure inside this directory matches the expected format:")
        print(f"{data_dir}/")
        print("  ├── train/")
        print("  │   ├── Normal/")
        print("  │   └── Lung Opacity/ (or other class names)")
        print("  └── test/")
        print("      ├── Normal/")
        print("      └── Lung Opacity/ (or other class names)")
        
    except Exception as e:
        # Error message printed within download_kaggle_dataset
        print(f"Script failed due to error: {e}")
        # Additional guidance already printed in download_kaggle_dataset

if __name__ == "__main__":
    main() 