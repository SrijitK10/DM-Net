import os
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

# ==== Configuration ====
MODEL_PATH = "./models/DM_NET_DCT_FINAL.keras"  # Path to your saved model
TEST_DIR = "/Users/srijit/Documents/Projects Personal/FREQUENCY/GANDCTAnalysis/processed_dct_log_scaled/test"  # Root directory containing 'real' and 'fake' subfolders
class_labels = {"0_real": 0, "1_fake": 1}  # Explicit label mapping

def preprocess_image(image):

    """Average Blur"""
    # kernel_size = (5, 5)
    # image = cv2.blur(image, kernel_size)

    """Gaussian Noise"""
    # sigma = 2.0
    # noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    # # Add the noise to the image
    # noisy_image = image + noise
    # # Clip the pixel values to the valid range [0, 255]
    # image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    """Median filter"""
    # size = 3
    # image = cv2.medianBlur(image, size)

    """Gamma correction"""
    # gamma = 1.2
    # image = image.astype(np.float32)
    # image /=255.0
    # image = np.power(image, gamma)
    # image = np.uint8(image * 255)

    """CLAHE"""
    # clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(3, 3))
    # b, g, r = cv2.split(image)
    # b_enhanced = clahe.apply(b)
    # g_enhanced = clahe.apply(g)
    # r_enhanced = clahe.apply(r)
    # image = cv2.merge([b_enhanced, g_enhanced, r_enhanced])



    return image
    

# ==== Load Dataset ====
def load_dataset(folder_path):
    images, labels = [], []
    print(f"Loading test dataset from: {folder_path}...")

    for folder in os.listdir(folder_path):
        if folder not in class_labels:
            continue
        label = class_labels[folder]
        subfolder_path = os.path.join(folder_path, folder)

        for file in os.listdir(subfolder_path):
            img_path = os.path.join(subfolder_path, file)
            # image = cv2.imread(img_path)
            image = np.load(img_path)
            if image is None:
                continue
            # image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)  # Bicubic interpolation
            images.append(image)
            labels.append(label)

    images = np.array(images, dtype='float32') # Normalize to [0, 1]
    labels = np.array(labels, dtype='int32')
    return images, labels

# ==== Evaluation ====
def evaluate_model():
    # Load model
    model = load_model(MODEL_PATH)

    # Load test data
    X_test, y_test = load_dataset(TEST_DIR)

    # Predict probabilities
    pred_probs = model.predict(X_test)
    pred_labels = (pred_probs > 0.5).astype(int).flatten()

    # Evaluation metrics
    accuracy = accuracy_score(y_test, pred_labels)
    auc = roc_auc_score(y_test, pred_probs)
    precision_fake = precision_score(y_test, pred_labels, pos_label=1)
    recall_fake = recall_score(y_test, pred_labels, pos_label=1)
    f1_fake = f1_score(y_test, pred_labels, pos_label=1)

    # Print results
    print("\n=== Evaluation Results ===")
    print(f"Accuracy         : {accuracy:.4f}")
    print(f"AUC              : {auc:.4f}")
    # print(f"Precision (Fake) : {precision_fake:.4f}")
    # print(f"Recall (Fake)    : {recall_fake:.4f}")
    # print(f"F1 Score (Fake)  : {f1_fake:.4f}")

# ==== Run ====
if __name__ == "__main__":
    evaluate_model()
