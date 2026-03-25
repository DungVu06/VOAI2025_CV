import torch
import cv2
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

class SignLanguageDataset(Dataset):
    def __init__(self, df, std_min=10, std_max=90, transform=None, target_frames=32):
        self.df = df
        self.std_min = std_min
        self.std_max = std_max
        self.transform = transform
        self.target_frames = target_frames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        video_path = self.df.iloc[idx]['file_path']
        label = self.df.iloc[idx]['label_id']
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                break
                    
            std_val = np.std(frame)
            if self.std_min < std_val < self.std_max:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transform:
                    transformed = self.transform(image=frame)
                    frame_tensor = transformed["image"]
                else:
                    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                frames.append(frame_tensor)
        
        cap.release()

        if len(frames) == 0:
            return torch.zeros((self.target_frames, 3, 224, 224)), label

        indices = np.linspace(0, len(frames) - 1, self.target_frames).astype(int)
        final_frames = [frames[i] for i in indices]
        
        return torch.stack(final_frames), label
    
if __name__ == "__main__":
    df = pd.read_csv("data/trainval.csv")
    dataset = SignLanguageDataset(df)
    frames, label_id = dataset[1]

    for i in range(frames.shape[0]):
        frame = frames[i].permute(1,2,0).numpy()
        frame = (frame * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(frame_bgr, f"Frame: {i+1}/32", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Video", frame_bgr)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()