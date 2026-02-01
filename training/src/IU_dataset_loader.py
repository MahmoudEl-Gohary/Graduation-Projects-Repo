import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path

class IndianaDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images" / "images_normalized"
        
        # Load and merge metadata
        reports = pd.read_csv(self.data_dir / "indiana_reports.csv")
        projections = pd.read_csv(self.data_dir / "indiana_projections.csv")
        
        # Merge on 'uid' and clean missing data
        self.data = pd.merge(projections, reports, on="uid", how="inner")
        self.data = self.data.dropna(subset=["findings", "impression"])
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 1. Load Image
        img_name = row["filename"]
        try:
            image = Image.open(self.images_dir / img_name).convert("RGB")
        except (FileNotFoundError, OSError):
            image = Image.new("RGB", (224, 224)) # Black image fallback

        # 2. Prepare Report
        findings = str(row.get("findings", ""))
        impression = str(row.get("impression", ""))
        full_report = f"Findings: {findings}\nImpression: {impression}"

        # 3. Return Raw Data
        return {
            "image": image, 
            "report": full_report,
            "filename": img_name
        }

if __name__ == "__main__":
    # Path setup
    REPO_ROOT = Path(__file__).resolve().parents[2]
    DATA_PATH = REPO_ROOT.parent / "data" / "indiana_university"
    
    if DATA_PATH.exists():
        dataset = IndianaDataset(DATA_PATH)
        print(f"Dataset Loaded. Size: {len(dataset)}")
        print(f"Sample: {dataset[0]['filename']}")
        print(f"Sample Report: {dataset[0]['report']}")
    else:
        print("Data path not found.")