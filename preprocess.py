import cv2
import numpy as np

def preprocess_image(image):

    """Average Blur"""
    kernel_size = (5, 5)
    image = cv2.blur(image, kernel_size)

    """Gaussian Noise"""
    sigma = 2.0
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    # Add the noise to the image
    noisy_image = image + noise
    # Clip the pixel values to the valid range [0, 255]
    image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    """Median filter"""
    size = 3
    image = cv2.medianBlur(image, size)

    """Gamma correction"""
    gamma = 1.2
    image = image.astype(np.float32)
    image /=255.0
    image = np.power(image, gamma)
    image = np.uint8(image * 255)

    """CLAHE"""
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(3, 3))
    b, g, r = cv2.split(image)
    b_enhanced = clahe.apply(b)
    g_enhanced = clahe.apply(g)
    r_enhanced = clahe.apply(r)
    image = cv2.merge([b_enhanced, g_enhanced, r_enhanced])