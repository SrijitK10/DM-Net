import torch

class Config:
    data_dir = "/Users/srijit/Documents/Projects Personal/FREQUENCY/GANDCTAnalysis/processed_dct_log_scaled/Protocol6_Mixed/3dgan" 
  # Update this
    batch_size = 32
    num_epochs = 20
    learning_rate = 1e-4
    checkpoint_path = "./models/swin_t_scratch_DCT.pth"
    num_classes = 2  # Binary classification
    device = torch.device("mps")
