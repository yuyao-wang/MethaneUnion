import torch.nn as nn
from torchvision import models


class FeatureRESNET18(nn.Module):
    def __init__(self, input_channel = 15, pretrained=False):
        super(FeatureRESNET18, self).__init__()
        self.net = models.resnet18(pretrained=pretrained)
        self.net.conv1 = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        return x

