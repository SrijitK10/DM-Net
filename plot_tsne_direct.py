import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import argparse
import os
from tqdm import tqdm

def plot_tsne_from_arrays(features_array, labels_array, output_path='tsne_direct_plot.png', 
                          perplexity=30, n_components=2):
    """
    Apply t-SNE on pre-extracted features and plot the results.
    
    Args:
        features_array: Path to numpy array containing features
        labels_array: Path to numpy array containing labels
        output_path: Path to save the plot
        perplexity: t-SNE perplexity parameter
        n_components: Number of dimensions for t-SNE (2 or 3)
    """
    # Load the arrays
    print(f"Loading features from {features_array}...")
    features = np.load(features_array)
    
    print(f"Loading labels from {labels_array}...")
    labels = np.load(labels_array)
    
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
    
    # Check if reshaping is needed
    if len(features.shape) > 2:
        features = features.reshape(features.shape[0], -1)
        print(f"Reshaped features to {features.shape}")
    
    # Apply PCA first to reduce dimensions (optional, but helps with performance)
    print("Applying PCA...")
    pca = PCA(n_components=min(50, features.shape[1]))
    features_pca = pca.fit_transform(features)
    
    # Apply t-SNE
    print(f"Applying t-SNE with perplexity={perplexity}...")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=1000, random_state=42)
    features_tsne = tsne.fit_transform(features_pca)
    
    # Plot the results
    plt.figure(figsize=(10, 8))
    
    if n_components == 2:
        # 2D plot
        scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], 
                     c=labels, alpha=0.6, cmap='coolwarm')
        plt.colorbar(scatter, label='Class')
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')
    else:
        # 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(features_tsne[:, 0], features_tsne[:, 1], features_tsne[:, 2], 
                  c=labels, alpha=0.6, cmap='coolwarm')
        plt.colorbar(scatter, label='Class')
        ax.set_xlabel('t-SNE dimension 1')
        ax.set_ylabel('t-SNE dimension 2')
        ax.set_zlabel('t-SNE dimension 3')
    
    plt.title('t-SNE visualization of feature arrays')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"t-SNE plot saved to {output_path}")
    plt.show()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Plot t-SNE visualization from numpy arrays')
    parser.add_argument('--features', type=str, required=True, 
                        help='Path to numpy array containing features')
    parser.add_argument('--labels', type=str, required=True,
                        help='Path to numpy array containing labels (0 for real, 1 for fake)')
    parser.add_argument('--perplexity', type=int, default=30,
                        help='t-SNE perplexity parameter')
    parser.add_argument('--dimensions', type=int, default=2, choices=[2, 3],
                        help='Number of dimensions for t-SNE (2 or 3)')
    parser.add_argument('--output', type=str, default='tsne_direct_plot.png',
                        help='Path to save the output plot')
    
    args = parser.parse_args()
    
    # Generate t-SNE plot
    plot_tsne_from_arrays(
        features_array=args.features,
        labels_array=args.labels,
        output_path=args.output,
        perplexity=args.perplexity,
        n_components=args.dimensions
    )

if __name__ == "__main__":
    main()
