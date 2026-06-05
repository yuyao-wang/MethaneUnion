import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> (h w) b c')

        # Apply MultiheadAttention
        attn_output, _ = self.attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Apply Feed Forward
        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        # Reshape back to (batch_size, channels, height, width)
        x = rearrange(x, '(h w) b c -> b c h w', h=h, w=w)
        return x
    
class HierarchicalTransformerEncoder(nn.Module):
    def __init__(self, in_channels, embed_dims, num_heads, depths, ff_dims):
        super(HierarchicalTransformerEncoder, self).__init__()
        self.stages = nn.ModuleList()
        self.embed_dims = embed_dims

        for i in range(len(depths)):
            stage = nn.Sequential(
                nn.Conv2d(in_channels if i == 0 else embed_dims[i-1], embed_dims[i], kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(1, embed_dims[i]),
                *[TransformerBlock(embed_dims[i], num_heads[i], ff_dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)

    def forward(self, x):
        features = []
        for stage in self.stages:
            # print(f'shape of x {x.shape}')
            x = stage(x)
            features.append(x)
        return features

class MLPDecoder(nn.Module):
    def __init__(self, embed_dims, num_classes):
        super(MLPDecoder, self).__init__()
        self.mlp_layers = nn.ModuleList()
        
        for dim in embed_dims:
            self.mlp_layers.append(nn.Sequential(
                nn.Conv2d(dim, num_classes, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ))

    def forward(self, features):
        seg_map = self.mlp_layers[0](features[0])
        for i in range(1, len(features)):
            seg_map += F.interpolate(self.mlp_layers[i](features[i]), size=seg_map.shape[2:], mode='bilinear', align_corners=False)
        return seg_map

class SegFormer(nn.Module):
    def __init__(self, in_channels, num_classes, embed_dims, num_heads, depths, ff_dims):
        super(SegFormer, self).__init__()
        self.encoder = HierarchicalTransformerEncoder(in_channels, embed_dims, num_heads, depths, ff_dims)
        self.decoder = MLPDecoder(embed_dims, num_classes)

    def forward(self, x):
        # print(f'shape of x {x.shape}')
        features = self.encoder(x)
        seg_map = self.decoder(features)
        return seg_map


from transformers import SegformerForSemanticSegmentation, SegformerConfig

class SegFormerWrapper(torch.nn.Module):
    def __init__(self, config):
        super(SegFormerWrapper, self).__init__()
        self.model = SegformerForSemanticSegmentation(config)

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        logits = outputs.logits
        # Translated comment
        input_height, input_width = pixel_values.shape[2:]  # Translated comment
        if logits.shape[-2:] != (input_height, input_width):
            logits = torch.nn.functional.interpolate(
                logits, size=(input_height, input_width), mode="bilinear", align_corners=False
            )
        return logits

def segformer(num_channels = 27, num_labels = 1):
    # Translated comment
    config = SegformerConfig(
        num_channels=num_channels,  # Translated comment
        num_labels=num_labels,  # Translated comment
        hidden_sizes=[32, 64, 160, 256],  # Translated comment
        depths=[2, 2, 2, 2],  # Translated comment
        patch_sizes=[7, 3, 3, 3],  # Translated comment
        strides=[4, 2, 2, 2],  # Translated comment
        decoder_hidden_size=256,  # Translated comment
        image_size=128
    )

    # initialize SegFormer model
    model = SegFormerWrapper(config)
    return model
