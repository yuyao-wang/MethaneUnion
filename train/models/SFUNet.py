import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F

# Attention Module
class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = torch.mean(x, dim=(2, 3), keepdim=True)  # Global average pooling
        attn = self.conv1(attn)
        attn = self.conv2(attn)
        attn = self.sigmoid(attn)
        return x * attn

# Fourier Transform Block
class FourierTransformBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(FourierTransformBlock, self).__init__()
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_real = nn.BatchNorm2d(out_channels)
        self.bn_imag = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x_fft = fft.fftn(x, dim=(-2, -1))
        real = x_fft.real
        imag = x_fft.imag

        real = self.conv_real(real)
        real = self.bn_real(real)
        imag = self.conv_imag(imag)
        imag = self.bn_imag(imag)

        x_fft_processed = torch.complex(real, imag)
        x_ifft = fft.ifftn(x_fft_processed, dim=(-2, -1))

        return x_ifft.real

# UNet Block with Attention-based Fusion
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super(UNetBlock, self).__init__()
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.frequency_branch = FourierTransformBlock(in_channels, out_channels)
        self.attention = Attention(out_channels)
        self.downsample = nn.MaxPool2d(2) if downsample else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        spatial_features = self.spatial_branch(x)
        frequency_features = self.frequency_branch(x)
        
        # Combine the two branches
        combined_features = spatial_features + frequency_features
        
        # Apply attention
        attn_features = self.attention(combined_features)
        
        # Downsample or upsample
        out = self.downsample(attn_features)
        out = self.relu(out)
        return attn_features, out

# UNet Decoder Block
class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        # 处理输入通道数为 out_channels * 2，因为拼接了 skip_connection
        self.block = UNetBlock(out_channels * 2, out_channels, downsample=False)

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        
        # 拼接 along the channel dimension
        x = torch.cat([x, skip_connection], dim=1)
        
        # 通过UNet block
        _, x = self.block(x)
        return x




# UNet Model
class SFUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SFUNet, self).__init__()
        self.encoder1 = UNetBlock(in_channels, 64)
        self.encoder2 = UNetBlock(64, 128)
        self.encoder3 = UNetBlock(128, 256)
        self.encoder4 = UNetBlock(256, 512)

        self.bottleneck = UNetBlock(512, 1024, downsample=False)

        self.decoder1 = UNetDecoderBlock(1024, 512)
        self.decoder2 = UNetDecoderBlock(512, 256)
        self.decoder3 = UNetDecoderBlock(256, 128)
        self.decoder4 = UNetDecoderBlock(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        f1, enc1 = self.encoder1(x)
        f2, enc2 = self.encoder2(enc1)
        f3, enc3 = self.encoder3(enc2)
        f4, enc4 = self.encoder4(enc3)

        # Bottleneck
        f5, bottleneck = self.bottleneck(enc4)

        # Decoder
        dec1 = self.decoder1(bottleneck, f4)
        dec2 = self.decoder2(dec1, f3)
        dec3 = self.decoder3(dec2, f2)
        dec4 = self.decoder4(dec3, f1)

        # Final Convolution
        out = self.final_conv(dec4)
        return out