import torch
import torch.nn as nn
import torch.nn.functional as F

class StochasticDepth(nn.Module):
    def __init__(self, drop_prob):
        super(StochasticDepth, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x / keep_prob * random_tensor
        return output

class WideFocus(nn.Module):
    def __init__(self, filters, dropout_rate):
        super(WideFocus, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(filters, filters, kernel_size=3, padding=3, dilation=3)
        self.conv_out = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x1 = self.dropout(F.gelu(self.conv1(x)))
        x2 = self.dropout(F.gelu(self.conv2(x)))
        x3 = self.dropout(F.gelu(self.conv3(x)))
        added = x1 + x2 + x3
        x_out = self.dropout(F.gelu(self.conv_out(added)))
        return x_out

class GlobalFilter(nn.Module):
    def __init__(self, dim, h, w):
        super(GlobalFilter, self).__init__()
        self.dim = dim
        self.h = h
        self.w = w
        self.weight = nn.Parameter(torch.randn(dim, h, w // 2 + 1, 2))

    def forward(self, x):
        B, a, b, C = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.fft.rfft2(x, dim=(-2, -1))
        weight = self.weight[..., 0] + 1j * self.weight[..., 1]
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b))
        x = x.permute(0, 2, 3, 1).contiguous()
        return x

class LocalFilter(nn.Module):
    def __init__(self, dim, h, w):
        super(LocalFilter, self).__init__()
        self.dim = dim
        self.h = h
        self.w = w
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(dim, h // 4, w // 2 // 4 + 1, 2)) for _ in range(16)
        ])

    def forward(self, x):
        B, a, b, C = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        blocks = [x[:, :, i*self.h//4:(i+1)*self.h//4, j*self.w//4:(j+1)*self.w//4] 
                  for i in range(4) for j in range(4)]
        fft_blocks = [torch.fft.rfft2(b, dim=(-2, -1)) for b in blocks]
        filtered_blocks = [fb * (w[..., 0] + 1j * w[..., 1]) for fb, w in zip(fft_blocks, self.weights)]
        ifft_blocks = [torch.fft.irfft2(fb, s=(self.h//4, self.w//4)) for fb in filtered_blocks]
        merged = torch.cat([torch.cat(ifft_blocks[i*4:(i+1)*4], dim=-1) for i in range(4)], dim=-2)
        x_out = merged.permute(0, 2, 3, 1).contiguous()
        return x_out

class GLFNetBlock(nn.Module):
    def __init__(self, dim, h, w):
        super(GLFNetBlock, self).__init__()
        self.global_filter = GlobalFilter(dim, h, w)
        self.local_filter = LocalFilter(dim, h, w)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.layer_norm = nn.LayerNorm([dim, h, w])
    
    def forward(self, x):
        x1 = self.global_filter(x)
        x2 = self.local_filter(x)
        x1 = F.relu(self.conv1(x1))
        x2 = F.relu(self.conv1(x2))
        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.conv2(x))
        x = self.layer_norm(x)
        return x

class GLFNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GLFNet, self).__init__()
        self.encoder1 = self.contracting_block(in_channels, 64)
        self.encoder2 = self.contracting_block(64, 128)
        self.encoder3 = self.contracting_block(128, 256)
        self.encoder4 = self.contracting_block(256, 512)
        
        self.bottleneck = self.contracting_block(512, 1024)
        
        self.upconv4 = self.expansive_block(1024, 512)
        self.upconv3 = self.expansive_block(1024, 256)
        self.upconv2 = self.expansive_block(512, 128)
        self.upconv1 = self.expansive_block(256, 64)
        
        self.final_layer = self.expansive_block(128, out_channels)
    
    def contracting_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return block
    
    def expansive_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        return block
    
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc4)
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((enc4, dec4), dim=1)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        
        out = self.final_layer(dec1)
        return out
