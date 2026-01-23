import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        
        self.shortcut = nn.Sequential()
        if input_dim != output_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class DeepMLP(nn.Module):
    def __init__(self):
        super(DeepMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128*128, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.dropout = nn.Dropout(0.5)
        
        self.resblock1 = ResidualBlock(4096, 1024)
        # self.resblock2 = ResidualBlock(2048, 1024)
        self.resblock3 = ResidualBlock(1024, 512)
        self.resblock4 = ResidualBlock(512, 256)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.bn1(self.fc1(x))
        x = self.dropout(x)
        
        x = self.resblock1(x)
        # x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        
        x = self.bn2(self.fc2(x))
        x = self.fc3(x)
        return x