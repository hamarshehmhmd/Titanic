#!/usr/bin/env python
"""
Titanic Dataset Downloader

This script downloads the Titanic dataset from Kaggle using the Kaggle API.
Before running, make sure you have:
1. Installed the Kaggle API: pip install kaggle
2. Set up your Kaggle API credentials:
   - Create a Kaggle account if you don't have one
   - Go to Account > Create API Token to download kaggle.json
   - Place kaggle.json in ~/.kaggle/ (or %USERPROFILE%\.kaggle\ on Windows)
   - Make sure permissions are set: chmod 600 ~/.kaggle/kaggle.json
"""

import os
import subprocess
import sys

def check_kaggle_api():
    """
    Check if the Kaggle API is installed and properly configured.
    """
    try:
        # Check if kaggle is installed
        subprocess.run(['kaggle', '--version'], 
                       stdout=subprocess.PIPE, 
                       stderr=subprocess.PIPE, 
                       check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Kaggle API not found. Please install it with: pip install kaggle")
        return False

def download_titanic_dataset():
    """
    Download the Titanic competition dataset from Kaggle.
    """
    if not check_kaggle_api():
        return False
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        # Download the Titanic dataset
        print("Downloading Titanic dataset from Kaggle...")
        subprocess.run(['kaggle', 'competitions', 'download', '-c', 'titanic', '-p', data_dir], 
                       stdout=subprocess.PIPE, 
                       check=True)
        
        # Unzip the dataset
        zip_file = os.path.join(data_dir, 'titanic.zip')
        if os.path.exists(zip_file):
            import zipfile
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            
            # Remove the zip file after extraction
            os.remove(zip_file)
        
        print(f"Titanic dataset successfully downloaded and extracted to {data_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading the dataset: {e}")
        print("Make sure your Kaggle API credentials are properly set up.")
        print("Visit https://github.com/Kaggle/kaggle-api for installation and configuration instructions.")
        return False

def manual_download_instructions():
    """
    Display instructions for manually downloading the dataset from Kaggle.
    """
    print("\nAlternatively, you can manually download the dataset:")
    print("1. Go to https://www.kaggle.com/c/titanic/data")
    print("2. Click the 'Download All' button (requires a Kaggle account)")
    print("3. Extract the downloaded ZIP file")
    print("4. Place train.csv, test.csv, and gender_submission.csv in the 'data' directory of this project\n")

if __name__ == "__main__":
    print("=" * 60)
    print("Titanic Dataset Downloader")
    print("=" * 60)
    
    success = download_titanic_dataset()
    
    if not success:
        manual_download_instructions()
        sys.exit(1)
    
    print("\nYou can now run the analysis scripts or Jupyter notebook.")
    print("=" * 60) 