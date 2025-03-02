import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset import get_dataloader
from model import SwinTransformerBinaryClassifier
from train import train_one_epoch, validate
from utils import save_checkpoint

# Set device
device = torch.device("mps")

def main():
    # Define dataset paths
    train_dir = "./datasets/train"
    val_dir = "./datasets/val"

    # Load Data
    train_loader, class_names = get_dataloader(train_dir, batch_size=32)
    val_loader, _ = get_dataloader(val_dir, batch_size=32, shuffle=False)

    # Initialize Model
    model = SwinTransformerBinaryClassifier().to(device)

    # Loss, Optimizer, Scheduler
    criterion = nn.BCELoss()  # Use BCELoss
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Training Loop
    num_epochs = 20
    best_val_acc = 0.0
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, best_val_acc, "best_checkpoint.pth")

        scheduler.step()
        print(f"📉 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"📈 Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # ---- Plot and Save ----
    plt.figure(figsize=(12, 5))

    # 📌 Training & Validation Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss", marker="o")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss", marker="s")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig("loss_plot.png")  # Save loss plot

    # 📌 Training & Validation Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label="Train Accuracy", marker="o")
    plt.plot(range(1, num_epochs + 1), val_accuracies, label="Validation Accuracy", marker="s")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig("accuracy_plot.png")  # Save accuracy plot

    plt.show()  # Show plots

# Run main function when script is executed
if __name__ == "__main__":
    main()