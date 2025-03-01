from train import train_model

# Define dataset paths
train_path = "./datasets/train"
val_path = "./datasets/val"

# Train the model
if __name__ == "__main__":
    model = train_model(train_path, val_path)
