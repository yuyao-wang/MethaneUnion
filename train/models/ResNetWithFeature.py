import torch
import torch.nn as nn
import torchvision.models as models

# Translated comment
class ResNetWithFeatures(nn.Module):
    def __init__(self, original_model, num_classes=2):
        super(ResNetWithFeatures, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])  # Translated comment
        
        # Translated comment
        self.features[0] = nn.Conv2d(18, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        self.pool = original_model.avgpool
        self.fc = nn.Linear(original_model.fc.in_features, num_classes)  # Translated comment

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        features = x.clone()  # Translated comment
        x = self.fc(x)
        return features, x

# Translated comment
original_model = models.resnet18(pretrained=True)  # Translated comment
model = ResNetWithFeatures(original_model, num_classes=2)

# Translated comment
input_tensor = torch.randn(1, 18, 96, 96)  # Translated comment
features, output = model(input_tensor)

print("Features shape:", features.shape)
print("Output shape:", output.shape)