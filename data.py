import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import os
from sklearn.utils import shuffle


def load_samples(csv_file):
    """Load image paths and labels from a CSV file."""
    data = pd.read_csv(csv_file)
    samples = list(zip(data['Path'], data['Truth']))
    return samples


def data_generator(samples, img_size=256, batch_size=32, num_classes=2, mode="train"):
    """
    General-purpose data generator for training, validation, and testing.
    Parameters:
        - samples: List of (image_path, label)
        - img_size: Target image size
        - batch_size: Batch size
        - num_classes: Number of output classes
        - mode: "train", "val", or "test"
    """
    num_samples = len(samples)
    
    while True:  # Infinite loop for generator
        if mode == "train":
            samples = shuffle(samples)  # Shuffle only for training
        
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset + batch_size]
            X_batch, y_batch = [], []

            for img_path, label in batch_samples:
                if not os.path.exists(img_path):
                    print(f"Warning: Missing image {img_path}")  # Log missing images
                    continue
                
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Error: Could not load {img_path}")  # Log unreadable images
                    continue
                
                img = cv2.resize(img, (img_size, img_size))
                # img = img.astype(np.float32) / 255.0  # Normalize
                
                X_batch.append(img)
                y_batch.append(label)

            if not X_batch:  # Skip if no valid images
                continue
            
            X_batch = np.array(X_batch)
            y_batch = tf.keras.utils.to_categorical(np.array(y_batch), num_classes=num_classes)

            yield X_batch, y_batch
