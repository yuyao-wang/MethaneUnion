import torch
import torch.nn as nn

class PANDA(nn.Module):
    def __init__(self, original_model):
        super(PANDA, self).__init__()
        self.features = nn.Sequential(*list(original_model.backbone.children())[:-2])
        self.fc1 = nn.Linear(512 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 1)  # 修改全连接层以适应二分类问题
        self.prototype = nn.Parameter(torch.randn(512))

        # self.features[0] = nn.Conv2d(18, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        self.pool = original_model.backbone.avgpool
        self.fc = original_model.backbone.fc

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        features = torch.flatten(x, 1)
        # proto_dist = torch.norm(features - self.prototype, dim=1, keepdim=True)
        # attention_weights = torch.sigmoid(proto_dist)
        output = self.fc(features)
        return features, output