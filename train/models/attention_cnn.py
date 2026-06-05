import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

class Attention(nn.Module):
    def __init__(self, in_dim):
        super(Attention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out

# Translated comment
class AttentionCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(AttentionCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=18, out_channels=64, kernel_size=3, padding=1)  # Translated comment
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.attention = Attention(in_dim=128)
        self.fc1 = nn.Linear(in_features=128 * 24 * 24, out_features=256)  # Translated comment
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, x):
        # print(f'input: {x.shape}')
        x = F.relu(self.conv1(x))
        # print(f'After conv1: {x.shape}')
        x = F.max_pool2d(x, 2)  # 48x48
        # print(f'After max_pool2d 1: {x.shape}')
        x = F.relu(self.conv2(x))
        # print(f'After conv2: {x.shape}')
        x = F.max_pool2d(x, 2)  # 24x24
        # print(f'After max_pool2d 2: {x.shape}')
        x = self.attention(x)
        # Translated comment
        x = x.view(x.size(0), -1)
        # Translated comment
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x