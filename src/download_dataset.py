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
        kaggle.api.dataset_download_files(dataset_name, path=output_dir, unzip=True) 
        print(f"Dataset downloaded and extracted successfully to {output_dir}")
    except Exception as e:
        print(f"Error during Kaggle API download: {e}")
        print("Please ensure your Kaggle API token is correctly set up in ~/.kaggle/kaggle.json")
        print("See README for setup instructions.")
        raise
    
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
            except zipfile.BadZipFile:
                print(f"Error: {zip_path} is not a valid zip file or is corrupted.")
            except Exception as e:
                print(f"Error unzipping {zip_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Download RSNA Pneumonia processed dataset from Kaggle')
    parser.add_argument('--dataset', type=str, default='iamtapendu/rsna-pneumonia-processed-dataset',
                        help='Kaggle dataset name (default: iamtapendu/rsna-pneumonia-processed-dataset)')
    parser.add_argument('--data-dir', type=str, default='./data/processed',
                        help='Directory to save the downloaded and extracted dataset (default: ./data/processed)')
    
    args = parser.parse_args()
    
    data_dir = os.path.abspath(args.data_dir)
    
    try:
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
        print(f"Script failed due to error: {e}")

if __name__ == "__main__":
    main() 