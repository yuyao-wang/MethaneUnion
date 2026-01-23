import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * out
    
class UNetEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, attention=False):
        super(UNetEncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.attention = ChannelAttention(out_channels) if attention else None

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        if self.attention:
            x = self.attention(x)
        return x

class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, attention=False):
        super(UNetDecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.attention = ChannelAttention(out_channels) if attention else None

    def forward(self, x, skip):
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        if self.attention:
            x = self.attention(x)
        return x

class ChannelAttentionUNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ChannelAttentionUNet, self).__init__()
        self.encoder1 = UNetEncoderBlock(in_channels, 64, attention=True)
        self.encoder2 = UNetEncoderBlock(64, 128, attention=True)
        self.encoder3 = UNetEncoderBlock(128, 256, attention=True)
        self.encoder4 = UNetEncoderBlock(256, 512, attention=True)

        self.bottleneck = UNetEncoderBlock(512, 1024, attention=True)

        self.decoder4 = UNetDecoderBlock(1024 + 512, 512, attention=True)
        self.decoder3 = UNetDecoderBlock(512 + 256, 256, attention=True)
        self.decoder2 = UNetDecoderBlock(256 + 128, 128, attention=True)
        self.decoder1 = UNetDecoderBlock(128 + 64, 64, attention=True)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(nn.functional.max_pool2d(enc1, 2))
        enc3 = self.encoder3(nn.functional.max_pool2d(enc2, 2))
        enc4 = self.encoder4(nn.functional.max_pool2d(enc3, 2))

        bottleneck = self.bottleneck(nn.functional.max_pool2d(enc4, 2))

        dec4 = self.decoder4(bottleneck, enc4)
        dec3 = self.decoder3(dec4, enc3)
        dec2 = self.decoder2(dec3, enc2)
        dec1 = self.decoder1(dec2, enc1)

        return self.final_conv(dec1)
