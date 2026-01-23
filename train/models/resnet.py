import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self, input_channel = 12, num_classes=2):
        super(ResNet18, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        
        self.backbone.conv1 = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        self.backbone.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(self.backbone.fc.in_features, num_classes))
        
    def forward(self, x):
        return self.backbone(x)
    
class ResNet34(nn.Module):
    def __init__(self, input_channel = 12, num_classes=2):
        super(ResNet34, self).__init__()
        self.backbone = models.resnet34(pretrained=False)
        
        self.backbone.conv1 = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        self.backbone.fc = nn.Sequential(nn.Dropout(0.5),
                                       nn.Linear(self.backbone.fc.in_features, num_classes))
        
    def forward(self, x):
        return self.backbone(x)

class ResNet50(nn.Module):
    def __init__(self, input_channel = 12, num_classes=2):
        super(ResNet50, self).__init__()
        self.backbone = models.resnet50(pretrained=False)
        
        self.backbone.conv1 = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        self.backbone.fc = nn.Sequential(nn.Dropout(0.5),
                                       nn.Linear(self.backbone.fc.in_features, num_classes))
        
    def forward(self, x):
        return self.backbone(x)
    
class ResNet101(nn.Module):
    def __init__(self, input_channel = 12, num_classes=2):
        super(ResNet101, self).__init__()
        self.backbone = models.resnet101(pretrained=False)
        
        self.backbone.conv1 = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        self.backbone.fc = nn.Sequential(nn.Dropout(0.5),
                                       nn.Linear(self.backbone.fc.in_features, num_classes))
        
    def forward(self, x):
        return self.backbone(x)