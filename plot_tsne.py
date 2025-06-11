import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
from keras.models import Model
from data import load_samples
import argparse
from tqdm import tqdm
import matplot2tikz

def load_data_arrays(numpy_arrays, labels):
    """
    Load a set of numpy arrays and their corresponding labels.
    
    Args:
        numpy_arrays: List of paths to numpy arrays
        labels: List of corresponding labels (0 for real, 1 for fake)
    
    Returns:
        X: Loaded numpy arrays
        y: Labels
    """
    X = []
    y = []
    
    for i, (array_path, label) in enumerate(tqdm(zip(numpy_arrays, labels), total=len(numpy_arrays))):
        try:
            # Load numpy array
            img = np.load(array_path)
            
            # Check for valid array
            if img is None or img.size == 0:
                print(f"Skipping empty array: {array_path}")
                continue
                
            if img.shape != (256, 256, 3):
                print(f"Skipping incorrect shape {img.shape}: {array_path}")
                continue
                
            # Convert data type to float32
            img = img.astype(np.float32)
            label = int(label)  # Ensure label is an integer
            
            X.append(img)
            y.append(label)
            
        except Exception as e:
            print(f"Error loading {array_path}: {e}")
            continue
    
    return np.array(X), np.array(y)

def extract_features(model, layer_name, data):
    """
    Extract features from a specific layer of the model.
    
    Args:
        model: Trained Keras model
        layer_name: Name of the layer to extract features from
        data: Input data to extract features from
    
    Returns:
        features: Extracted features
    """
    # For sequential model, we need to create a new model with specified input and output
    try:
        # First, ensure the model has been built by passing a sample through it
        sample_data = data[0:1]  # Take first sample as a dummy input
        _ = model(sample_data)  # This builds the model
        
        # Now create a feature extraction model
        feature_model = Model(inputs=model.input, 
                             outputs=model.get_layer(layer_name).output)
    except Exception as e:
        print(f"Error creating feature model: {e}")
        print("Trying alternative approach...")
        
        # Alternative approach: create intermediate model
        feature_model = tf.keras.Sequential()
        
        # Copy layers up to the target layer
        found_layer = False
        for layer in model.layers:
            feature_model.add(layer)
            if layer.name == layer_name:
                found_layer = True
                break
        
        if not found_layer:
            print(f"Layer '{layer_name}' not found. Available layers:")
            for i, layer in enumerate(model.layers):
                print(f"{i}: {layer.name}")
            raise ValueError(f"Layer '{layer_name}' not found in model")
    
    # Extract features in batches to avoid memory issues
    batch_size = 32
    features = []
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        batch_features = feature_model.predict(batch, verbose=0)
        features.append(batch_features)
    
    return np.vstack(features)

def plot_tsne(features, labels, output_path='tsne_plot.png', perplexity=30, n_components=2):
    """
    Apply t-SNE and plot the results.
    
    Args:
        features: Extracted features
        labels: Corresponding labels
        output_path: Path to save the plot
        perplexity: t-SNE perplexity parameter
        n_components: Number of dimensions for t-SNE (2 or 3)
    """
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
        # 2D plot with discrete classes instead of continuous color bar
        # Create a scatter plot for each class
        real_mask = labels == 0
        fake_mask = labels == 1
        
        plt.scatter(features_tsne[real_mask, 0], features_tsne[real_mask, 1], 
                   alpha=0.6, c="#6df5bf", label='Real')
        plt.scatter(features_tsne[fake_mask, 0], features_tsne[fake_mask, 1], 
                   alpha=0.6, c='#97a2ff', label='Fake')
        
        plt.legend(title='Class')
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')
        matplot2tikz.save("test.tex")
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
    
    plt.title('t-SNE visualization')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"t-SNE plot saved to {output_path}")
    plt.show()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Plot t-SNE visualization for model features')
    parser.add_argument('--model', type=str, required=True, 
                        help='Path to the trained model (.keras file)')
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to CSV file containing image paths and labels')
    parser.add_argument('--layer', type=str, default='flatten',
                        help='Name of the layer to extract features from')
    parser.add_argument('--perplexity', type=int, default=30,
                        help='t-SNE perplexity parameter')
    parser.add_argument('--dimensions', type=int, default=2, choices=[2, 3],
                        help='Number of dimensions for t-SNE (2 or 3)')
    parser.add_argument('--output', type=str, default='tsne_plot.png',
                        help='Path to save the output plot')
    parser.add_argument('--max_samples', type=int, default=1000,
                        help='Maximum number of samples to use (to avoid memory issues)')
    parser.add_argument('--list_layers', action='store_true',
                        help='List all layer names in the model and exit')
    
    args = parser.parse_args()
    
    # Load the model
    print(f"Loading model from {args.model}...")
    model = tf.keras.models.load_model(args.model)
    model.summary()
    
    # If --list_layers is specified, print all layer names and exit
    if args.list_layers:
        print("\nAvailable layers:")
        for i, layer in enumerate(model.layers):
            print(f"{i}: {layer.name} (type: {layer.__class__.__name__}, shape: {layer.output_shape if hasattr(layer, 'output_shape') else 'unknown'})")
        return
    
    # Load sample data from CSV
    print(f"Loading samples from {args.csv}...")
    samples = load_samples(args.csv)
    
    # Limit the number of samples if needed
    if len(samples) > args.max_samples:
        print(f"Limiting to {args.max_samples} samples...")
        samples = samples[:args.max_samples]
    
    # Split paths and labels
    numpy_arrays = [path for path, _ in samples]
    labels = [label for _, label in samples]
    
    # Load the data
    print("Loading data...")
    X, y = load_data_arrays(numpy_arrays, labels)
    print(f"Loaded {len(X)} samples with shape {X.shape}")
    
    # Extract features
    print(f"Extracting features from layer '{args.layer}'...")
    features = extract_features(model, args.layer, X)
    print(f"Extracted features with shape {features.shape}")
    
    # Reshape features if needed
    if len(features.shape) > 2:
        features = features.reshape(features.shape[0], -1)
        print(f"Reshaped features to {features.shape}")
    
    # Plot t-SNE
    plot_tsne(features, y, 
              output_path=args.output,
              perplexity=args.perplexity,
              n_components=args.dimensions)

if __name__ == "__main__":
    main()