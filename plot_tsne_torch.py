import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import argparse
import os
from tqdm import tqdm
from model import SwinTinyBinary
from config import Config
from from_numpy import NumpyDataset
from torch.utils.data import DataLoader
import timm

def load_data_from_dataset(data_dir, max_samples=1000):
    """
    Load data using the NumpyDataset class.
    
    Args:
        data_dir: Directory containing the data in the format expected by NumpyDataset
        max_samples: Maximum number of samples to load
    
    Returns:
        features: Loaded numpy arrays
        labels: Corresponding labels
    """
    print(f"Loading data from {data_dir} using NumpyDataset...")
    dataset = NumpyDataset(data_dir)
    
    # Limit the number of samples if needed
    if len(dataset) > max_samples:
        print(f"Limiting to {max_samples} samples...")
        indices = np.random.choice(len(dataset), max_samples, replace=False)
        # Create a subset of the dataset
        data_subset = [dataset[i] for i in indices]
    else:
        data_subset = [dataset[i] for i in range(len(dataset))]
    
    # Extract features and labels
    features = []
    labels = []
    
    for image, label in tqdm(data_subset, desc="Loading data"):
        # Convert from PyTorch tensor (C, H, W) back to numpy (H, W, C)
        features.append(image.permute(1, 2, 0).numpy())
        labels.append(label.item())
    
    return np.array(features), np.array(labels)

def extract_features_from_model(model, numpy_arrays, layer_name=None, device='cuda'):
    """
    Extract features from a specific layer of the model.
    
    Args:
        model: PyTorch model
        numpy_arrays: Numpy arrays to extract features from
        layer_name: Name of the layer to extract features from (not used for Swin)
        device: Device to run the model on
    
    Returns:
        features: Extracted features
    """
    # Set model to evaluation mode
    model.eval()
    
    # Convert numpy arrays to PyTorch tensors with the correct shape (N, C, H, W)
    tensors = []
    for img in numpy_arrays:
        # Convert from (H, W, C) to (C, H, W)
        tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        tensors.append(tensor)
    
    tensors = torch.stack(tensors).to(device)
    
    # Extract features in batches to avoid memory issues
    batch_size = 32
    features = []
    
    with torch.no_grad():
        for i in range(0, len(tensors), batch_size):
            batch = tensors[i:i+batch_size]
            
            # For Swin Transformer, we can extract features by modifying the forward method
            try:
                # Try to use the forward_features method if available
                batch_features = model.model.forward_features(batch)
            except AttributeError:
                # Fallback: get features from the penultimate layer
                print("forward_features not available, using alternative approach")
                # This is a bit of a hack - we're assuming the model is a Swin model with a head at the end
                # Run the model up to the last layer but not including it
                batch_features = model.model.forward_features(batch)
            
            # Convert to numpy and flatten if needed
            batch_features = batch_features.cpu().numpy()
            features.append(batch_features)
    
    # Concatenate all batches
    features = np.vstack(features)
    return features

def plot_tsne(features, labels=None, output_path='tsne_plot.png', perplexity=30, n_components=2):
    """
    Apply t-SNE and plot the results.
    
    Args:
        features: Features to visualize
        labels: Corresponding labels (optional)
        output_path: Path to save the plot
        perplexity: t-SNE perplexity parameter
        n_components: Number of dimensions for t-SNE (2 or 3)
    """
    # Check if reshaping is needed
    if len(features.shape) > 2:
        features = features.reshape(features.shape[0], -1)
        print(f"Reshaped features to {features.shape}")
    
    # Apply PCA first to reduce dimensions
    print("Applying PCA...")
    pca = PCA(n_components=min(50, features.shape[1]))
    features_pca = pca.fit_transform(features)
    
    # Apply t-SNE
    print(f"Applying t-SNE with perplexity={perplexity}...")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=1000, random_state=42)
    features_tsne = tsne.fit_transform(features_pca)
    
    # Plot the results
    plt.figure(figsize=(10, 8))
    
    if labels is not None:
        # Plot with class labels
        if n_components == 2:
            # 2D plot with discrete classes
            real_mask = labels == 0
            fake_mask = labels == 1
            
            plt.scatter(features_tsne[real_mask, 0], features_tsne[real_mask, 1], 
                       alpha=0.6, c='blue', label='Real')
            plt.scatter(features_tsne[fake_mask, 0], features_tsne[fake_mask, 1], 
                       alpha=0.6, c='red', label='Fake')
            
            plt.legend(title='Class')
            plt.xlabel('t-SNE dimension 1')
            plt.ylabel('t-SNE dimension 2')
        else:
            # 3D plot with discrete classes
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            real_mask = labels == 0
            fake_mask = labels == 1
            
            ax.scatter(features_tsne[real_mask, 0], features_tsne[real_mask, 1], features_tsne[real_mask, 2],
                      alpha=0.6, c='blue', label='Real')
            ax.scatter(features_tsne[fake_mask, 0], features_tsne[fake_mask, 1], features_tsne[fake_mask, 2],
                      alpha=0.6, c='red', label='Fake')
            
            ax.legend(title='Class')
            ax.set_xlabel('t-SNE dimension 1')
            ax.set_ylabel('t-SNE dimension 2')
            ax.set_zlabel('t-SNE dimension 3')
    else:
        # Plot without class labels
        if n_components == 2:
            plt.scatter(features_tsne[:, 0], features_tsne[:, 1], alpha=0.6)
            plt.xlabel('t-SNE dimension 1')
            plt.ylabel('t-SNE dimension 2')
        else:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(features_tsne[:, 0], features_tsne[:, 1], features_tsne[:, 2], alpha=0.6)
            ax.set_xlabel('t-SNE dimension 1')
            ax.set_ylabel('t-SNE dimension 2')
            ax.set_zlabel('t-SNE dimension 3')
    
    plt.title('t-SNE visualization')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"t-SNE plot saved to {output_path}")
    plt.show()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Plot t-SNE visualization for PyTorch model features')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Directory containing the data in format expected by NumpyDataset (e.g., "datasets/test")')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to the PyTorch model (.pth file)')
    parser.add_argument('--perplexity', type=int, default=30,
                        help='t-SNE perplexity parameter')
    parser.add_argument('--dimensions', type=int, default=2, choices=[2, 3],
                        help='Number of dimensions for t-SNE (2 or 3)')
    parser.add_argument('--output', type=str, default='tsne_plot.png',
                        help='Path to save the output plot')
    parser.add_argument('--max_samples', type=int, default=1000,
                        help='Maximum number of samples to use (to avoid memory issues)')
    
    args = parser.parse_args()
    
    # Load data using the NumpyDataset class
    features, labels = load_data_from_dataset(args.data_dir, args.max_samples)
    print(f"Loaded {len(features)} samples with shape {features.shape}")
    
    # Check if we need to extract features from raw images using the model
    if args.model is not None:
        # Determine device
        device = torch.device('mps')
        
        # Load the model
        model = SwinTinyBinary()
        model.load_state_dict(torch.load(args.model, map_location=device))
        model = model.to(device)
        
        # Extract features
        print("Extracting features from model...")
        features = extract_features_from_model(model, features, device=device)
    
    # Plot t-SNE
    print("Plotting t-SNE...")
    plot_tsne(features, labels, 
              output_path=args.output,
              perplexity=args.perplexity,
              n_components=args.dimensions)

if __name__ == "__main__":
    main()
