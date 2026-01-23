import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.2):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob)
        )

    def forward(self, x):
        return self.conv(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(batchsize, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batchsize, C, width, height)
        out = self.gamma*out + x
        return out

class CSAClassifier(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super(CSAClassifier, self).__init__()
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = ConvBlock(512, 1024)
        self.self_attention = SelfAttention(1024)
        
        self.channel_attention1 = ChannelAttention(64)
        self.channel_attention2 = ChannelAttention(128)
        self.channel_attention3 = ChannelAttention(256)
        self.channel_attention4 = ChannelAttention(512)
        
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        enc1 = self.channel_attention1(self.enc1(x))
        enc2 = self.channel_attention2(self.enc2(self.pool(enc1)))
        enc3 = self.channel_attention3(self.enc3(self.pool(enc2)))
        enc4 = self.channel_attention4(self.enc4(self.pool(enc3)))
        
        bottleneck = self.bottleneck(self.pool(enc4))
        bottleneck = self.self_attention(bottleneck)
        
        # 平均池化并展平
        bottleneck = F.adaptive_avg_pool2d(bottleneck, 1).view(bottleneck.size(0), -1)
        out = self.fc(bottleneck)
        
        return out
