import torch
from config import Config
from model import SwinTinyBinary
from data_loader import get_dataloaders

def test_model():
    _, _, test_loader = get_dataloaders()
    model = SwinTinyBinary().to(Config.device)
    model.load_state_dict(torch.load(Config.checkpoint_path, map_location=Config.device))
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(Config.device), labels.to(Config.device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")