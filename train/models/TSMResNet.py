import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

# Temporal Shift Module
class TemporalShift(nn.Module):
    def __init__(self, n_segment=3, n_div=8):
        super(TemporalShift, self).__init__()
        self.n_segment = n_segment  # Number of frames
        self.fold_div = n_div
    
    def forward(self, x):
        # Assuming input is (batch_size * n_segment, c, h, w)
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)  # Reshape to (batch_size, n_segment, channels, h, w)
        
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        # Shift some channels forward
        out[:, :-1, :fold] = x[:, 1:, :fold]  # Shift forward
        # Shift some channels backward
        out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]  # Shift backward
        # Keep the rest unchanged
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
        
        return out.view(nt, c, h, w)  # Reshape back to (batch_size * n_segment, c, h, w)

# ResNet18 with TSM integrated for temporal modeling
class ResNet18TSM(nn.Module):
    def __init__(self, n_segment=3, num_channels=10, num_classes = 1):
        super(ResNet18TSM, self).__init__()
        self.n_segment = n_segment
        
        # Load ResNet18 model
        self.model = resnet18(pretrained=False)
        
        # Modify the first convolutional layer to accept 10 channels instead of 3
        self.model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Add Temporal Shift Modules before each block
        self.shift1 = TemporalShift(n_segment=self.n_segment)
        self.shift2 = TemporalShift(n_segment=self.n_segment)
        self.shift3 = TemporalShift(n_segment=self.n_segment)
        self.shift4 = TemporalShift(n_segment=self.n_segment)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        # Input shape: (batch_size, n_segment, num_channels, height, width)
        b, c, h, w = x.size()
        x = x.view(b * self.n_segment, c // self.n_segment, h, w)  # Merge batch and time dimensions (b * t, c, h, w)
        
        # Apply temporal shift before each stage
        x = self.shift1(x)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.shift2(x)
        x = self.model.layer1(x)
        
        x = self.shift3(x)
        x = self.model.layer2(x)
        
        x = self.shift4(x)
        x = self.model.layer3(x)
        
        x = self.model.layer4(x)
        
        # Average pooling and fully connected layer
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        
        # Reshape back to (batch_size, n_segment, num_classes)
        x = x.view(b, self.n_segment, -1)
        return x.mean(dim=1)  # Average over time (frames)

# Example usage
# model = ResNet18TSM(n_segment=3, num_channels=10)
# input_tensor = torch.randn(2, 3, 10, 128, 128)  # (batch_size, n_segment, num_channels, height, width)
# output = model(input_tensor)
# print(output.shape)  # Output shape: (batch_size, num_classes)
