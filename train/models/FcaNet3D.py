import torch.nn as nn
import torch
import math

def get_freq_indices(method):
    # 频域选择方法
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiSpectralAttentionLayer3D(nn.Module):
    def __init__(self, channel, dct_d, dct_h, dct_w, reduction=16, freq_sel_method='top16'):
        super(MultiSpectralAttentionLayer3D, self).__init__()
        self.reduction = reduction
        self.dct_d = dct_d
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]

        self.dct_layer = MultiSpectralDCTLayer3D(dct_d, dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, d, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w or d != self.dct_d:
            x_pooled = nn.functional.adaptive_avg_pool3d(x, (self.dct_d, self.dct_h, self.dct_w))
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1, 1)
        return x * y.expand_as(x)

class MultiSpectralDCTLayer3D(nn.Module):
    def __init__(self, depth, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer3D, self).__init__()
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)
        self.register_buffer('weight', self.get_dct_filter(depth, height, width, mapper_x, mapper_y, channel))

    def forward(self, x):
        assert len(x.shape) == 5, 'x must be 5-dimensional, but got ' + str(len(x.shape))
        x = x * self.weight
        result = torch.sum(x, dim=[2, 3, 4])  # 在D, H, W三个维度上进行求和
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, depth, height, width, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, depth, height, width)
        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_d in range(depth):
                for t_x in range(height):
                    for t_y in range(width):
                        dct_filter[i * c_part: (i+1)*c_part, t_d, t_x, t_y] = (
                            self.build_filter(t_x, u_x, height) *
                            self.build_filter(t_y, v_y, width)
                        )
        return dct_filter


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class FcaBasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 reduction=16):
        super(FcaBasicBlock3D, self).__init__()
        c2dwh = dict([(64, (16, 64, 64)), (128, (8, 32, 32)), (256, (4, 16, 16)), (512, (2, 8, 8))])
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.att = MultiSpectralAttentionLayer3D(planes, *c2dwh[planes], reduction=reduction, freq_sel_method='top16')
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.att(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet3D(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet3D, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(10, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 输入通道为10
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feature_map = self.layer4(x)

        x = self.avgpool(feature_map)
        x = torch.flatten(x, 1)
        logits = self.fc(x)

        return feature_map, logits

def fcanet3d18(num_classes=1000, input_channel=10, pretrained=False):
    """Constructs a FCA-Net3D-18 model."""
    model = ResNet3D(FcaBasicBlock3D, [2, 2, 2, 2], num_classes=num_classes)
    
    # 修改第一层的输入通道数，适应 (frames, channels, D, 128, 128) 的输入
    model.conv1 = nn.Conv3d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.bn1 = nn.BatchNorm3d(64)
    model.relu = nn.ReLU(inplace=True)
    model.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
    
    model.avgpool = nn.AdaptiveAvgPool3d(1)
    return model

# 生成随机输入 (batch_size, frames, channels, depth, height, width)
input_tensor = torch.randn(8, 3, 10, 32, 128, 128)  # batch_size=8, frames=10, channels=10, depth=32, height=128, width=128

# 初始化模型
model = fcanet3d18(num_classes=2, input_channel=10)

# 运行模型前向传播
output = model(input_tensor)

print("Output shape:", output.shape)