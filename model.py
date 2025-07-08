# import torch.nn as nn
# import timm
# from config import Config

# class SwinTinyBinary(nn.Module):
#     def __init__(self):
#         super(SwinTinyBinary, self).__init__()
#         self.model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=Config.num_classes)

#     def forward(self, x):
#         return self.model(x)
    
import torch
import torch.nn as nn
import timm
from config import Config

class SwinTinyBinary(nn.Module):
    def __init__(self):
        super(SwinTinyBinary, self).__init__()
        self.model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=Config.num_classes)

    def forward(self, x):
        # Compute 2D FFT
        x_fft = torch.fft.fft2(x, dim=(-2, -1))  # Apply FFT along height and width
        #x_fft = torch.fft.fftshift(x_fft, dim=(-2, -1))  # Shift zero-frequency component to the center
        x_mag = torch.abs(x_fft)  # Get magnitude spectrum

        return self.model(x_mag)
