import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import cv2
from scipy import fftpack


# def dct2(array):
#     array = fftpack.dct(array, type=2, norm="ortho", axis=0)
#     array = fftpack.dct(array, type=2, norm="ortho", axis=1)
#     return array

def load_samples(csv_file):
    """Load image paths and labels from a CSV file."""
    data = pd.read_csv(csv_file)
    samples = list(zip(data['Path'], data['Truth']))
    return samples

def data_generator(samples,  batch_size=32, shuffle_data=True, num_classes=2):
    """
    Yields batches of images and labels for training.
    """
    num_samples = len(samples)
    
    while True:  # Infinite generator loop
        if shuffle_data:
            samples = shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset + batch_size]

            X_train = []
            y_train = []

            for img_path, label in batch_samples:
                try:
                    # Load image 
                    # img = cv2.imread(img_path)
                    # img = cv2.resize(img, (256,256))
                    # # img= dct2(img)
                    img = np.load(img_path)

                    

                    # Debugging: Check for invalid images
                    if img is None or img.size == 0:
                        print(f"Skipping empty image: {img_path}")
                        continue

                    if img.shape != (256, 256, 3):
                        print(f"Skipping incorrect shape {img.shape}: {img_path}")
                        continue

                    # Convert data type to float32 and normalize
                    img = img.astype(np.float32)  

                    X_train.append(img)
                    y_train.append(label)

                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue  # Skip problematic images

            if not X_train:
                continue  # Skip empty batch

            # Convert to NumPy arrays
            X_train = np.array(X_train, dtype=np.float32)  
            y_train = np.array(y_train, dtype=np.int32)

            # One-hot encode labels
            y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)

            yield X_train, y_train
