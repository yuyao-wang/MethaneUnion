import torch
import torch.nn as nn

import torch
from torch import nn
from transformers import ViTConfig, ViTModel

class VisionTransformer(nn.Module):
    def __init__(self, num_channels = 1, image_size = 32, num_classes = 1, patch_size = 4, hidden_size = 192, num_attention_heads = 3, intermediate_size = 768):
        super(VisionTransformer, self).__init__()
        print(f'vit hidden_size: {hidden_size}, num_attention_heads: {num_attention_heads}, image_size: {image_size} patch_size: {patch_size}')
        config = ViTConfig()
        config.hidden_size = hidden_size
        config.num_attention_heads = num_attention_heads
        config.num_hidden_layers = 12
        config.image_size = image_size
        config.num_channels = num_channels
        config.patch_size = patch_size  # 或者根据您的需求调整
        config.num_labels = num_classes
        config.intermediate_size = intermediate_size

        self.vit = ViTModel(config)
        self.classifier = nn.Linear(config.hidden_size, num_classes)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        output = self.classifier(outputs.last_hidden_state[:, 0])
        return output