import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=16, stride=16)
        
    def forward(self, x):
        x = self.proj(x)  # Convert to patches
        x = x.flatten(2).transpose(1, 2)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=2, stride=2)
        
    def forward(self, x, skip=None):
        print(f'shape of x {x.shape} shape of skip {skip.shape}')
        if skip is not None:
            x = self.upsample(x)
            x = torch.cat([x, skip], dim=1)
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        return x

class TransformerUNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(TransformerUNet, self).__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim=512)
        self.transformer = nn.Sequential(
            TransformerBlock(embed_dim=512, num_heads=8, mlp_dim=2048),
            TransformerBlock(embed_dim=512, num_heads=8, mlp_dim=2048),
            TransformerBlock(embed_dim=512, num_heads=8, mlp_dim=2048),
            TransformerBlock(embed_dim=512, num_heads=8, mlp_dim=2048)
        )
        
        self.encoder_conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.encoder_norm1 = nn.BatchNorm2d(64)
        self.encoder_relu1 = nn.ReLU()
        
        self.encoder_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.encoder_norm2 = nn.BatchNorm2d(64)
        self.encoder_relu2 = nn.ReLU()
        
        self.encoder_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.encoder_norm3 = nn.BatchNorm2d(128)
        self.encoder_relu3 = nn.ReLU()
        
        self.encoder_conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.encoder_norm4 = nn.BatchNorm2d(256)
        self.encoder_relu4 = nn.ReLU()
        
        self.decoder4 = DecoderBlock(1024, 512)
        self.decoder3 = DecoderBlock(768, 256)
        self.decoder2 = DecoderBlock(384, 128)
        self.decoder1 = DecoderBlock(192, 64)
        
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        enc1 = self.encoder_relu1(self.encoder_norm1(self.encoder_conv1(x)))
        enc2 = self.encoder_relu2(self.encoder_norm2(self.encoder_conv2(enc1)))
        enc3 = self.encoder_relu3(self.encoder_norm3(self.encoder_conv3(enc2)))
        enc4 = self.encoder_relu4(self.encoder_norm4(self.encoder_conv4(enc3)))

        x = self.patch_embed(x)
        x = x.permute(1, 0, 2)  # Rearrange for transformer input
        x = self.transformer(x)
        x = x.permute(1, 2, 0).view(x.size(1), 512, 8, 8)  # Reshape back to image dimensions

        d4 = self.decoder4(x, enc4)
        d3 = self.decoder3(d4, enc3)
        d2 = self.decoder2(d3, enc2)
        d1 = self.decoder1(d2, enc1)
        
        out = self.sigmoid(self.final_conv(d1))
        return out