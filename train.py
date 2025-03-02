import torch
from tqdm import tqdm
from utils import calculate_accuracy

# Training function
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    loop = tqdm(dataloader, total=len(dataloader))

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)  # Ensure labels are [batch_size, 1]

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (outputs > 0.5).eq(labels).sum().item()  # Apply threshold to probabilities
        total += labels.size(0)

        loop.set_description(f"Train Loss: {loss.item():.4f}")

    return running_loss / len(dataloader), correct / total * 100

# Validation function
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)  # Ensure labels are [batch_size, 1]
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            correct += (outputs > 0.5).eq(labels).sum().item()  # Apply threshold to probabilities
            total += labels.size(0)

    return running_loss / len(dataloader), correct / total * 100