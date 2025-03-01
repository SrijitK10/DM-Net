import torch
import torch.nn as nn

class LPQNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(LPQNet, self).__init__()

        self.convRes = nn.Conv2d(in_channels, 32, kernel_size=3, padding=0)
        
        self.conv1 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)  # Adjust stride for correct output size
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(128)

        # Flatten layer: dynamically get the input size
        self.flatten_size = self._get_conv_output_size(in_channels)

        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)

    def _get_conv_output_size(self, in_channels):
        """ Helper function to determine the correct input size for the FC layer """
        with torch.no_grad():
            x = torch.randn(1, in_channels, 256, 256)  # Assuming input size is 64x64
            x = self.convRes(x)
            x = self.pool1(self.bn1(self.conv1(x)))
            x = self.pool2(self.bn2(self.conv3(self.conv2(x))))
            x = self.pool3(self.bn3(self.conv5(self.conv4(x))))
            return x.view(1, -1).size(1)  # Flattened size

    def forward(self, x):
        x = self.convRes(x)
        x = self.pool1(self.bn1(self.conv1(x)))
        x = self.pool2(self.bn2(self.conv3(self.conv2(x))))
        x = self.pool3(self.bn3(self.conv5(self.conv4(x))))

        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))

        return x
