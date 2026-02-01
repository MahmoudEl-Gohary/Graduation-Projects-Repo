import os
import sys
from pathlib import Path

# --- CONFIGURATION ---
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = REPO_ROOT.parent / "data" / "indiana_university"
KAGGLE_DATASET = "raddar/chest-xrays-indiana-university"

def setup_directories():
    if not DATA_ROOT.exists():
        print(f"Creating data directory: {DATA_ROOT}")
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Data directory exists: {DATA_ROOT}")

def download_dataset():
    kaggle_config = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_config.exists():
        print("Error: ~/.kaggle/kaggle.json not found.")
        print("Please create an API token in your Kaggle settings and upload it to a [.kaggle] folder in the ROOT folder.")
        return False

    print(f"Downloading {KAGGLE_DATASET}...")
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        # Download and Unzip automatically
        api.dataset_download_files(
            KAGGLE_DATASET, 
            path=DATA_ROOT, 
            unzip=True,
            quiet=False
        )
        print("Download and extraction complete.")
        return True
        
    except ImportError:
        print("Error: 'kaggle' library not found.")
        return False
    except Exception as e:
        print(f"Download failed: {e}")
        return False

def verify_files():
    images_dir = DATA_ROOT / "images" / "images_normalized"
    reports_csv = DATA_ROOT / "indiana_reports.csv"
    
    if images_dir.exists() and reports_csv.exists():
        print("Data verification successful!")
        count = len([f for f in os.listdir(images_dir) if f.endswith('.png')])
        print(f"   - Found {count} X-ray images.")
        return True
    else:
        print("Warning: Files missing. Check the folder structure.")
        return False

if __name__ == "__main__":
    setup_directories()
    if download_dataset():
        verify_files()