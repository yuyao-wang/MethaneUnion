import torch
import torch.nn as nn

class DoubleConv3D(nn.Module):
    """Translated to English."""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels):
        super(UNet3D, self).__init__()

        # Translated comment
        self.enc1 = DoubleConv3D(in_channels, 64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.enc2 = DoubleConv3D(64, 128)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.enc3 = DoubleConv3D(128, 256)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # Bottleneck
        self.bottleneck = DoubleConv3D(256, 512)

        # Translated comment
        self.upconv3 = nn.ConvTranspose3d(512, 256, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec3 = DoubleConv3D(512, 256)
        self.upconv2 = nn.ConvTranspose3d(256, 128, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec2 = DoubleConv3D(256, 128)
        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = DoubleConv3D(128, 64)

        # Translated comment
        self.out_conv = nn.Conv3d(64, 1, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, -1, 3, H, W)
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3))

        # Decoder
        dec3 = self.dec3(torch.cat((self.upconv3(bottleneck), enc3), dim=1))
        dec2 = self.dec2(torch.cat((self.upconv2(dec3), enc2), dim=1))
        dec1 = self.dec1(torch.cat((self.upconv1(dec2), enc1), dim=1))

        # outputlayer
        out = self.out_conv(dec1)  # Translated comment

        # Translated comment
        out = torch.mean(out, dim=2)  # (batch_size, 1, 1, 32, 32)

        return out

# Translated comment
# model = UNet3D(in_channels=9)
# x = torch.randn(1, 9, 3, 32, 32)  # (batch_size, channels, frames, height, width)
# output = model(x)

# Translated comment
