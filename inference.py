import torch
from torchvision import transforms
from model import SwinBinaryClassifier
from PIL import Image

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SwinBinaryClassifier().to(device)
model.load_state_dict(torch.load("best_swin_model.pth"))
model.eval()

# Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Prediction Function
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        prob = torch.sigmoid(output).item()

    class_label = "REAL" if prob > 0.5 else "FAKE"
    print(f"Predicted: {class_label} ({prob:.2f})")

# Example Usage
predict("/path/to/test_image.jpg")
