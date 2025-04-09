import os

def check_create_dir(directory):
    """
    Check if a directory exists and create it if it doesn't.
    
    Args:
        directory (str): Path to the directory to check/create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    return directory
