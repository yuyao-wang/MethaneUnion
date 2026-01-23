import torch
import torch.nn as nn
import torchvision.models as models

# 定义一个新的ResNet类，继承自预训练的ResNet
class ResNetWithFeatures(nn.Module):
    def __init__(self, original_model, num_classes=2):
        super(ResNetWithFeatures, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])  # 提取除了最后两个层的所有层
        
        # 修改第一个卷积层以适应输入通道数18
        self.features[0] = nn.Conv2d(18, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        self.pool = original_model.avgpool
        self.fc = nn.Linear(original_model.fc.in_features, num_classes)  # 修改全连接层以适应二分类问题

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        features = x.clone()  # 克隆特征图，以防修改影响后续操作
        x = self.fc(x)
        return features, x

# 加载预训练的ResNet模型
original_model = models.resnet18(pretrained=True)  # 可以选择不同的ResNet版本，例如resnet50
model = ResNetWithFeatures(original_model, num_classes=2)

# 测试模型
input_tensor = torch.randn(1, 18, 96, 96)  # 假设输入大小为96x96的18通道图像
features, output = model(input_tensor)

print("Features shape:", features.shape)
print("Output shape:", output.shape)