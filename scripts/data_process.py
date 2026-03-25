import pickle
import numpy as np
import pandas as pd
import unicodedata

from sklearn.model_selection import train_test_split
from pathlib import Path

def normalize_text(text):
    return unicodedata.normalize("NFC", text)

def create_csv_file(root, label_map):
    root = Path(root)
    data = []

    for label_folder in root.iterdir():
        if label_folder.is_dir():
            label_name = normalize_text(label_folder.name)
            label_id = label_map.get(label_name, -1)
            video_files = list(label_folder.glob("*.mp4"))

            for video_path in video_files:
                filename = video_path.name
                if any(suffix in filename for suffix in ["_1.mp4", "_2.mp4", "_3.mp4"]):
                    continue

                data.append({
                    "file_path": str(video_path),
                    "label_name": label_name,
                    "label_id": label_id
                })
    
    df = pd.DataFrame(data)
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label_id"]
    )
    df.loc[train_df.index, "split"] = "train"
    df.loc[val_df.index, "split"] = "val"
    df.to_csv("data/trainval.csv", index=False)

if __name__ == "__main__":
    with open("data/label_mapping.pkl", "rb") as f:
        label_map = pickle.load(f)
    label_map = {normalize_text(k): v for k, v in label_map.items()}

    create_csv_file("data/train", label_map)