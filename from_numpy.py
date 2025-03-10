import torch
import numpy as np
import os
from torch.utils.data import Dataset
    

class NumpyDataset(Dataset):
    def __init__(self, root_dir):
        self.data = []
        self.labels = []
        
        for class_label, class_name in enumerate(["0_real", "1_fake"]):
            class_path = os.path.join(root_dir, class_name)
            for file_name in os.listdir(class_path):
                if file_name.endswith('.npy'):
                    file_path = os.path.join(class_path, file_name)
                    self.data.append(file_path)
                    self.labels.append(class_label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx]
        label = self.labels[idx]

        # Load NumPy array
        image = np.load(file_path).astype(np.float32)  # Ensure float32 format

        # Normalize (optional, depends on model)
        # image = image / 255.0  # Scale between 0 and 1

        # Convert to PyTorch tensor (H, W, C) → (C, H, W)
        image = torch.tensor(image).permute(2, 0, 1)  # Change shape to (3, 256, 256)

        label = torch.tensor(label, dtype=torch.long)  # Convert label to tensor

        return image, label