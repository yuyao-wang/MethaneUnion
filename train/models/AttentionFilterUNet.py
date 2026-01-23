import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # print(f'shape of g1 {g1.shape} shape of x1 {x1.shape}')
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class FilterGate(nn.Module):
    def __init__(self, in_channels):
        super(FilterGate, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        gate = self.gate(x)
        return x * gate

class FourierTransformModule(nn.Module):
    def __init__(self):
        super(FourierTransformModule, self).__init__()
    
    def forward(self, x):
        # 对输入特征图进行傅里叶变换
        fft = torch.fft.fft2(x)
        # 处理频域信息（如应用滤波器或增强特定频率成分）
        # 在这里可以添加自定义的频域操作
        return fft
    
    def inverse(self, x):
        # 逆傅里叶变换回到空间域
        ifft = torch.fft.ifft2(x)
        return ifft.real

class FourierAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(FourierAttentionBlock, self).__init__()
        self.fourier_transform = FourierTransformModule()
        self.attention_real = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.attention_imag = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 转换到频域
        freq_x = self.fourier_transform(x)
        
        # 将复数分为实部和虚部
        real_x = freq_x.real
        imag_x = freq_x.imag
        
        # 分别对实部和虚部应用注意力机制
        real_attention = self.attention_real(real_x)
        imag_attention = self.attention_imag(imag_x)
        
        # 组合实部和虚部
        combined_attention = torch.complex(real_attention, imag_attention)
        
        # 逆傅里叶变换回到空间域
        combined_attention = self.fourier_transform.inverse(combined_attention)
        
        return x * combined_attention.real

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 下采样块1
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # 下采样块2
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # 下采样块3
        self.enc4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # 下采样块4
        self.enc5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        return e1, e2, e3, e4, e5

class Decoder(nn.Module):
    def __init__(self, out_channels):
        super(Decoder, self).__init__()
        
        # 上采样块1
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # 上采样块2
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # 上采样块3
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 最终输出层
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, e4, e3, e2, e1):
        d1 = self.up1(e4)
        d1 = torch.cat([d1, e3], dim=1)  # 跳跃连接，拼接特征图
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d3 = self.up3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        d3 = self.dec3(d3)
        
        out = self.final(d3)  # 最终输出
        
        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, skip_connection):
        # x = self.upconv(x)
        x = torch.cat([x, skip_connection], dim=1)  # 跳跃连接
        x = self.conv_block(x)
        return x

class Bottleneck(nn.Module):
    def __init__(self):
        super(Bottleneck, self).__init__()
        self.pool = nn.MaxPool2d(2)  # 进一步池化
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x = self.pool(x)
        return self.bottleneck(x)

class AttentionFilterUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionFilterUNet, self).__init__()
        
        self.encoder = Encoder(in_channels)
        self.bottleneck = Bottleneck()
        
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder1 = DecoderBlock(1024, 512)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder2 = DecoderBlock(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = DecoderBlock(256, 128)
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder4 = DecoderBlock(128, 64)
        # self.upconv5 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        # 频域注意力块
        self.fourier_attention1 = FourierAttentionBlock(512)
        self.fourier_attention2 = FourierAttentionBlock(256)
        self.fourier_attention3 = FourierAttentionBlock(128)
        self.fourier_attention4 = FourierAttentionBlock(64)
        
        # 空间域注意力块
        self.spatial_attention1 = AttentionBlock(F_g=512, F_l=512, F_int=512)
        self.spatial_attention2 = AttentionBlock(F_g=256, F_l=256, F_int=256)
        self.spatial_attention3 = AttentionBlock(F_g=128, F_l=128, F_int=128)
        self.spatial_attention4 = AttentionBlock(F_g=64, F_l=64, F_int=64)

        # 过滤门
        self.filter_gate1 = FilterGate(512)
        self.filter_gate2 = FilterGate(256)
        self.filter_gate3 = FilterGate(128)
        self.filter_gate4 = FilterGate(64)

        # 最终输出层
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # 编码器部分
        e1, e2, e3, e4, e5 = self.encoder(x)
        
        # Bottleneck部分
        # bottleneck = self.bottleneck(e4)
        
        # 频域注意力机制与解码器
        up1 = self.upconv1(e5)
        freq_bottleneck = self.fourier_attention1(up1)
        # print(f'shape of freq_bottleneck {freq_bottleneck.shape}, shape of e4 {e4.shape}')
        d1 = self.decoder1(up1, e4)
        
        # print(f'shape of d1 {d1.shape}')
        up2 = self.upconv2(d1)
        freq_d1 = self.fourier_attention2(up2)
        # print(f'shape of freq_d1 {freq_d1.shape}, shape of e3 {e3.shape}')
        d2 = self.decoder2(freq_d1, e3)

        up3 = self.upconv3(d2)
        freq_d2 = self.fourier_attention3(up3)
        # print(f'shape of freq_d2 {freq_d2.shape}, shape of e2 {e2.shape}')
        d3 = self.decoder3(freq_d2, e2)

        up4 = self.upconv4(d3)
        freq_d3 = self.fourier_attention4(up4)
        # print(f'shape of freq_d3 {freq_d3.shape}, shape of e1 {e1.shape}')
        d4 = self.decoder4(freq_d3, e1)

        # print(f'shape of d4 {d4.shape}')
        # 空间域注意力机制
        spatial_d1 = self.spatial_attention1(up1, e4)
        spatial_d2 = self.spatial_attention2(up2, e3)
        spatial_d3 = self.spatial_attention3(up3, e2)
        spatial_d4 = self.spatial_attention4(up4, e1)

        # print(f'shape of spatial_d1 {spatial_d4.shape}')
        # 将频域和空间域的结果融合
        combined_d1 = self.filter_gate1(d1 + spatial_d1)
        combined_d2 = self.filter_gate2(d2 + spatial_d2)
        combined_d3 = self.filter_gate3(d3 + spatial_d3)
        combined_d4 = self.filter_gate4(d4 + spatial_d4)

        # print(f'shape of spatial_d1 {combined_d4.shape}')
        # 最终输出
        out = self.final(combined_d4)
        return out