from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_data_loaders(train_dir, val_dir, batch_size=32, num_workers=1):
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_data = datasets.ImageFolder(root=val_dir, transform=val_transform)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, val_loader
