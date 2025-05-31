import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from config import Config
from model import SwinTinyBinary
from data_loader import get_dataloaders

def test_model():
    test_loader = get_dataloaders()
    
    # Load the trained model
    model = SwinTinyBinary().to(Config.device)
    model.load_state_dict(torch.load(Config.checkpoint_path, map_location=Config.device))
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []  # Stores probabilities for ROC-AUC calculation

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(Config.device), labels.to(Config.device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Get probabilities for the positive class
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert lists to NumPy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="binary")
    recall = recall_score(all_labels, all_preds, average="binary")
    f1 = f1_score(all_labels, all_preds, average="binary")
    roc_auc = roc_auc_score(all_labels, all_probs)

    print("\nTest Results:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["REAL", "FAKE"]))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, ["REAL", "FAKE"])

    # ROC Curve (saved as PNG)
    plot_roc_curve(all_labels, all_probs, save_path="roc_curve.png")

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png", dpi=300)
    print("Confusion matrix saved as confusion_matrix.png")

def plot_roc_curve(y_true, y_scores, save_path="roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc_score(y_true, y_scores):.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random guess line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid()
    
    # Save the ROC curve as a PNG file
    plt.savefig(save_path, dpi=300)
    print(f"ROC curve saved as {save_path}")

