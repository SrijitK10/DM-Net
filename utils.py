import torch

# Save the best model checkpoint
def save_checkpoint(model, optimizer, epoch, best_val_acc, filename="best_swin_model.pth"):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
    }
    torch.save(checkpoint, filename)
    print(f"✅ Model Saved! Best Validation Accuracy: {best_val_acc:.2f}%")

# Calculate Accuracy
def calculate_accuracy(outputs, labels):
    preds = outputs > 0.5  # Apply threshold to probabilities
    correct = (preds == labels).sum().item()
    return correct / labels.size(0) * 100