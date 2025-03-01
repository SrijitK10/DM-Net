import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import LPQNet
from train import train, validate, plot_results

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Load datasets
    train_data = datasets.ImageFolder(root="./datasets/train", transform=train_transform)
    val_data = datasets.ImageFolder(root="./datasets/val", transform=val_transform)

    # DataLoader
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)  # num_workers=0 for Windows
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=0)

    # Model, loss, optimizer
    model = LPQNet(in_channels=3, num_classes=2).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 30
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train(train_loader, model, loss_fn, optimizer, device, epoch)
        validate(val_loader, model, loss_fn, device, epoch)

    print("Training complete!")
    plot_results()

if __name__ == "__main__":  # Required for Windows multiprocessing
    main()
