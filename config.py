import torch

class Config:
    data_dir = "./processed_dct_log_scaled"  # Update this
    batch_size = 32
    num_epochs = 20
    learning_rate = 1e-4
    checkpoint_path = "best_swin_tiny.pth"
    num_classes = 2  # Binary classification
    device = torch.device("mps" if torch.cuda.is_available() else "cpu")
