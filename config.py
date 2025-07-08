import torch

class Config:
    data_dir = "./deepfake-eval" 
  # Update this
    batch_size = 32
    num_epochs = 20
    learning_rate = 1e-4
    checkpoint_path = "./models/best_swin_tiny_scratch_raw.pth"
    num_classes = 2  # Binary classification
    device = torch.device("mps")
