import torch.nn as nn
import torch.nn.functional as F
from efficient_kan import KAN

class CNNRegression(nn.Module):
    def __init__(self):
        super(CNNRegression, self).__init__()
        # Input size: [batch_size, 1, 128, 128]
        self.conv1 = nn.Conv2d(4, 16, kernel_size=5, stride=1, padding=2)  # Output size: [batch_size, 16, 128, 128]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Output size: [batch_size, 16, 64, 64]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)  # Output size: [batch_size, 32, 64, 64]
        # Second pooling layer output size: [batch_size, 32, 32, 32]
        self.fc1 = nn.Linear(32 * 32 * 32, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)  # Output a single value
        self.kan_layer = KAN([32 * 32 * 32, 64, 1])

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)  # Flatten the tensor for the fully connected layer
        # x = F.leaky_relu(self.fc1(x))
        # x = F.leaky_relu(self.fc2(x))
        # x = self.fc3(x)  # No activation function, as we're outputting a single regression value
        x = self.kan_layer(x)
        return x