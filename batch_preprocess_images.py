import os
import cv2
import numpy as np
from preprocess import preprocess_image
import argparse

# Individual preprocessing functions

def average_blur(image):
    kernel_size = (5, 5)
    return cv2.blur(image, kernel_size)

def gaussian_noise(image):
    sigma = 2.0
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def median_filter(image):
    size = 5
    return cv2.medianBlur(image, size)

def gamma_correction(image):
    gamma = 1.2
    image = image.astype(np.float32)
    image /= 255.0
    image = np.power(image, gamma)
    return np.uint8(image * 255)

def clahe_enhance(image):
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(3, 3))
    b, g, r = cv2.split(image)
    b_enhanced = clahe.apply(b)
    g_enhanced = clahe.apply(g)
    r_enhanced = clahe.apply(r)
    return cv2.merge([b_enhanced, g_enhanced, r_enhanced])

# Map step names to functions
preprocessing_steps = {
    'average_blur': average_blur,
    'gaussian_noise': gaussian_noise,
    'median_filter': median_filter,
    'gamma_correction': gamma_correction,
    'clahe': clahe_enhance,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a single preprocessing method to all images in a folder.")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input folder with images')
    parser.add_argument('--method', type=str, required=True, choices=list(preprocessing_steps.keys()), help='Preprocessing method to apply')
    args = parser.parse_args()

    INPUT_DIR = args.input_dir
    step_name = args.method
    func = preprocessing_steps[step_name]
    output_dir = f'output_{step_name}'
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(INPUT_DIR, filename)
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.png')
            image = cv2.imread(input_path)
            if image is None:
                print(f"Failed to load {input_path}")
                continue
            processed = func(image)
            if processed is None:
                print(f"Preprocessing failed for {input_path} at {step_name}")
                continue
            cv2.imwrite(output_path, processed)
            print(f"Saved: {output_path}")
