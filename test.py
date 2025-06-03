import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from sklearn.utils import shuffle
import os
import argparse

from data import load_samples, data_generator


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

batch_size = 32

def fft_layer(x):
    x = tf.signal.fft2d(tf.cast(x, tf.complex64))
    return tf.abs(x)

# Load the model with the custom function

def main():
    # Define image shape based on input type
    img_size = (256,256)
    img_shape = (img_size,img_size,3)

    
    batch_size = 32
    test_data_path = "./csv/test.csv"  # Path to the training data

    test_samples = load_samples(test_data_path)

    print(len(test_samples))

    test_generator = data_generator(
        test_samples, batch_size,
    )

    # Load the model
    # model = tf.keras.models.load_model("./models/DM_NET_DFT2.keras", custom_objects={"fft_layer": fft_layer})
    model = tf.keras.models.load_model("./models/DM_NET_DCT2.keras")
    batch_size = 32

    # Evaluate the model
    
    results = model.evaluate(test_generator, batch_size=batch_size, verbose=1,steps = len(test_samples))
    print("Test loss:", results[0])
    print("Test accuracy:", results[1])


if __name__ == '__main__':
    main()
