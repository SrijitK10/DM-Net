import tensorflow as tf
import numpy as np
import cv2

# Example grayscale image (random values)
image = cv2.imread('/Users/srijit/Documents/Projects Personal/FREQUENCY/GANDCTAnalysis/datasets/train/1_fake/adm1.png')
print("Shape of the image: ", image.type)
# Convert to complex
image_complex = tf.cast(image, tf.complex64)

# Compute 2D FFT
fft_image = tf.signal.fft2d(image_complex)
#shape of the image
print("Shape of the image: ", tf.shape(fft_image))

# print("Reconstructed Image:\n", tf.math.real(reconstructed_image).numpy())
