import torch
import torch.nn as nn
import timm

class SwinTransformerBinaryClassifier(nn.Module):
    def __init__(self, model_name='swin_tiny_patch4_window7_224', pretrained=True, num_classes=1):
        super(SwinTransformerBinaryClassifier, self).__init__()
        
        # Load the Swin Transformer model with ImageNet weights
        self.swin_transformer = timm.create_model(model_name, pretrained=pretrained)
        
        # Remove the original classifier head
        in_features = self.swin_transformer.head.in_features
        self.swin_transformer.head = nn.Identity()  # Remove the final classification layer
        
        # Add a global average pooling layer to reduce spatial dimensions
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Add a new fully connected layer for binary classification
        self.fc = nn.Linear(in_features, num_classes)
        
        # Add a sigmoid activation function for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Forward pass through the Swin Transformer
        x = self.swin_transformer.forward_features(x)  # Extract features
        
        # Apply global average pooling to reduce spatial dimensions
        x = self.global_pool(x)  # Shape: [batch_size, in_features, 1, 1]
        x = x.flatten(1)  # Flatten to [batch_size, in_features]
        
        # Pass through the fully connected layer
        x = self.fc(x)  # Shape: [batch_size, num_classes]
        
        # Apply sigmoid activation to get probabilities
        x = self.sigmoid(x)
        
        return x

# Example usage
if __name__ == "__main__":
    # Create the model
    model = SwinTransformerBinaryClassifier()
    
    # Print the model architecture
    print(model)
    
    # Example input tensor (batch_size, channels, height, width)
    example_input = torch.randn(32, 3, 224, 224)  # Batch size of 32
    
    # Forward pass
    output = model(example_input)
    print(output.shape)  # Expected output shape: [32, 1]