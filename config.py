import torch

class Config:
    data_dir = "./gamma1.2" 
  # Update this
    batch_size = 32
    num_epochs = 20
    learning_rate = 1e-4
    checkpoint_path = "./models/Best Swin Tiny FFT.pth"
    num_classes = 2  # Binary classification
    device = torch.device("mps")
