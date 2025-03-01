import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bar

train_accuracies, val_accuracies = [], []
train_losses, val_losses = [], []
best_val_acc = 0.0  # Track the best validation accuracy

def train(dataloader, model, loss_fn, optimizer, device, epoch):
    model.train()
    total_loss, correct = 0, 0
    num_batches, size = len(dataloader), len(dataloader.dataset)

    # Progress bar for training loop
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1} [Training]", leave=False)

    for X, y in loop:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (pred.argmax(1) == y).sum().item()

        # Update tqdm description with current loss
        loop.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / num_batches
    accuracy = 100 * correct / size
    train_accuracies.append(accuracy)
    train_losses.append(avg_loss)

    print(f"Train Accuracy: {accuracy:.2f}%, Train Loss: {avg_loss:.4f}")

def validate(dataloader, model, loss_fn, device, epoch):
    global best_val_acc
    model.eval()
    total_loss, correct = 0, 0
    num_batches, size = len(dataloader), len(dataloader.dataset)

    # Progress bar for validation loop
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1} [Validation]", leave=False)

    with torch.no_grad():
        for X, y in loop:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            total_loss += loss.item()
            correct += (pred.argmax(1) == y).sum().item()

            # Update tqdm description with current loss
            loop.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / num_batches
    accuracy = 100 * correct / size
    val_accuracies.append(accuracy)
    val_losses.append(avg_loss)

    print(f"Validation Accuracy: {accuracy:.2f}%, Validation Loss: {avg_loss:.4f}")

    # Save model if it achieves highest validation accuracy
    if accuracy > best_val_acc:
        best_val_acc = accuracy
        torch.save(model.state_dict(), f"best_model_epoch_{epoch+1}.pth")
        print(f"New best model saved with accuracy: {accuracy:.2f}%")

def plot_results():
    epochs = range(1, len(train_accuracies) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Training & Validation Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Train vs Validation Accuracy")

    # Plot Training & Validation Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Train vs Validation Loss")

    plt.show()
