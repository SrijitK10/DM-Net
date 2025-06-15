import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from config import Config
from from_numpy import NumpyDataset

def get_dataloaders():
    

    # train_dataset = NumpyDataset(os.path.join(Config.data_dir, "train"))
    # val_dataset = NumpyDataset(os.path.join(Config.data_dir, "val"))
    test_dataset = NumpyDataset(os.path.join(Config.data_dir, "test"))

    # train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4)

    return test_loader

# import os
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# from torch.utils.data import DataLoader
# from config import Config

# def get_dataloaders():
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#     ])

#     # train_dataset = datasets.ImageFolder(os.path.join(Config.data_dir, "train"), transform=transform)
#     # val_dataset = datasets.ImageFolder(os.path.join(Config.data_dir, "val"), transform=transform)
#     test_dataset = datasets.ImageFolder(os.path.join(Config.data_dir, "test"), transform=transform)

#     # train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4)
#     # val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4)
#     test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4)

#     return test_loader





