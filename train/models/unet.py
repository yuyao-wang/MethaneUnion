import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)   
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
class GroupConv3D(nn.Module):
    def __init__(self, in_channels=27, group_out_channels=6, final_out_channels=64):
        super(GroupConv3D, self).__init__()
        
        self.pre_conv3d = nn.Conv3d(in_channels=3, out_channels=3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.group_out_channels = group_out_channels
        self.relu = nn.ReLU(inplace=True)

        # unimportance_out = 6
        # essential_out = 8
        self.conv1 = nn.Conv2d(7, group_out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(1, group_out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(1, group_out_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(7, group_out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(1, group_out_channels, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(1, group_out_channels, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(7, group_out_channels, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(1, group_out_channels, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(1, group_out_channels, kernel_size=3, padding=1)

        self.post_conv3d = nn.Conv3d(in_channels=3, out_channels=3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        
        # 汇总卷积层
        self.final_conv = nn.Conv2d(group_out_channels * 9, final_out_channels, kernel_size=3, padding=1)
        # self.final_conv = nn.Identity()

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, 3, 9, height, width)
        x = self.pre_conv3d(x)

        x = x.view(batch_size, -1, height, width)

        group1 = x[:, :7, :, :]
        group2 = x[:, 7:8, :, :]
        group3 = x[:, 8:9, :, :]
        group4 = x[:, 9:16, :, :]
        group5 = x[:, 16:17, :, :]
        group6 = x[:, 17:18, :, :]
        group7 = x[:, 18:25, :, :]
        group8 = x[:, 25:26, :, :]
        group9 = x[:, 26:27, :, :]
        
        # 分别应用卷积
        out1 = self.conv1(group1)
        out2 = self.conv2(group2)
        out3 = self.conv3(group3)
        out4 = self.conv4(group4)
        out5 = self.conv5(group5)
        out6 = self.conv6(group6)
        out7 = self.conv7(group7)
        out8 = self.conv8(group8)
        out9 = self.conv9(group9)
        
        concatenated = torch.cat([out1, out2, out3, out4, out5, out6, out7, out8, out9], dim=1)
        
        # print(f'shape concatenated {concatenated.shape}')
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, 3, -1, height, width)
        x = self.post_conv3d(x)

        x = x.view(batch_size, -1, height, width)

        final_out = self.final_conv(concatenated)
        
        return final_out

class MEECUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(MEECUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = GroupConv3D()
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)   
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
class UNetSimple(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNetSimple, self).__init__()
        
        # 编码部分 (两次下采样)
        self.inc = DoubleConv(n_channels, 64)  # 输入卷积
        self.down1 = nn.MaxPool2d(2)           # 第一次下采样
        self.conv1 = DoubleConv(64, 128)       # 卷积

        self.down2 = nn.MaxPool2d(2)           # 第二次下采样
        self.conv2 = DoubleConv(128, 256)      # 卷积

        # 解码部分 (上采样)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 第一次上采样
        self.conv3 = DoubleConv(256, 128)  # 合并卷积

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)   # 第二次上采样
        self.conv4 = DoubleConv(128, 64)   # 合并卷积

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)  # 输出层

    def forward(self, x):
        # 编码路径
        x1 = self.inc(x)       # (batch, 64, 32, 32)
        x2 = self.down1(x1)    # (batch, 64, 16, 16)
        x2 = self.conv1(x2)    # (batch, 128, 16, 16)

        x3 = self.down2(x2)    # (batch, 128, 8, 8)
        x3 = self.conv2(x3)    # (batch, 256, 8, 8)

        # 解码路径
        x = self.up1(x3)       # (batch, 128, 16, 16)
        x = torch.cat([x2, x], dim=1)  # 跳跃连接 (skip connection)
        x = self.conv3(x)      # (batch, 128, 16, 16)

        x = self.up2(x)        # (batch, 64, 32, 32)
        x = torch.cat([x1, x], dim=1)  # 跳跃连接 (skip connection)
        x = self.conv4(x)      # (batch, 64, 32, 32)

        logits = self.outc(x)  # (batch, n_classes, 32, 32)
        return logits



class MTLUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(MTLUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.up_recon1 = Up(1024, 512 // factor, bilinear)
        self.up_recon2 = Up(512, 256 // factor, bilinear)
        self.up_recon3 = Up(256, 128 // factor, bilinear)
        self.up_recon4 = Up(128, 64, bilinear)
        self.out_recon = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)   
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        x_recon = self.up_recon1(x5, x4)
        x_recon = self.up2(x_recon, x3)
        x_recon = self.up3(x_recon, x2)
        x_recon = self.up4(x_recon, x1)
        recon = self.out_recon(x_recon)
        return logits, recon