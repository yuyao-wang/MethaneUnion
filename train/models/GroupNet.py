import torch.nn as nn
import torch
import torch.nn.functional as F
# from torchvision.models import ResNet
import math

def get_freq_indices(method):
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

class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction = 16, freq_sel_method = 'top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x] 
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n,c,h,w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered. 
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()
        
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2,3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
    
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
                        
        return dct_filter


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class FcaBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(FcaBasicBlock, self).__init__()
        # 根据输入尺寸 128x128 重新计算后的 c2wh
        c2wh = dict([(64, 64), (128, 32), (256, 16), (512, 8)])
        self.planes = planes
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.att = MultiSpectralAttentionLayer(planes, c2wh[planes], c2wh[planes], reduction=reduction, freq_sel_method='top16')
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

# def fcanet18(num_classes=1000, input_channel = 10, pretrained=False):
#     """Constructs a FcaNet-18 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(FcaBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    
#     # 修改第一层的输入通道数，适应 (10, 128, 128) 的输入
#     model.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
#     model.bn1 = nn.BatchNorm2d(64)
#     model.relu = nn.ReLU(inplace=True)
#     model.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
#     model.avgpool = nn.AdaptiveAvgPool2d(1)
#     return model

class GroupFCAResNet(nn.Module):
    def __init__(self, input_channels, block, layers, num_classes=1000):
        super(GroupFCAResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = GroupConv()
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 第一部分的特征提取
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 各层的特征提取
        x = self.layer1(x)
        x = self.layer2(x)  # 假设在 layer2 之后提取特征
        x = self.layer3(x)
        feature_map = self.layer4(x)

        # 分类头
        x = self.avgpool(feature_map)
        x = torch.flatten(x, 1)
        logits = self.fc(x)

        # 返回特征和分类结果
        return logits
    
class GroupFCAResNet3D(nn.Module):
    def __init__(self, input_channels, block, layers, num_classes=1000):
        super(GroupFCAResNet3D, self).__init__()
        self.inplanes = 64
        self.conv1 = GroupConv3D()
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 第一部分的特征提取
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 各层的特征提取
        x = self.layer1(x)
        x = self.layer2(x)  # 假设在 layer2 之后提取特征
        x = self.layer3(x)
        feature_map = self.layer4(x)

        # 分类头
        x = self.avgpool(feature_map)
        x = torch.flatten(x, 1)
        logits = self.fc(x)

        # 返回特征和分类结果
        return logits
# 定义基本残差块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out
    
class GroupConv(nn.Module):
    def __init__(self, in_channels=27, group_out_channels=8, final_out_channels=64):
        super(GroupConv, self).__init__()
        
        # 每组卷积输出通道数
        self.group_out_channels = group_out_channels
        
        # 定义每组的卷积层
        self.conv1 = nn.Conv2d(7, group_out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(1, group_out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(1, group_out_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(7, group_out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(1, group_out_channels, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(1, group_out_channels, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(7, group_out_channels, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(1, group_out_channels, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(1, group_out_channels, kernel_size=3, padding=1)
        
        # 汇总卷积层
        self.final_conv = nn.Conv2d(group_out_channels * 9, final_out_channels, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
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
        out4 = self.conv4(group4) # 单通道组
        out5 = self.conv5(group5)
        out6 = self.conv6(group6)
        out7 = self.conv7(group7) # 单通道组
        out8 = self.conv8(group8)
        out9 = self.conv9(group9) 
        
        # 拼接结果
        concatenated = torch.cat([out1, out2, out3, out4, out5, out6, out7, out8, out9], dim=1)
        
        # 汇总卷积
        final_out = self.final_conv(concatenated)
        
        return final_out
    
class GroupConv3DRelu(nn.Module):
    def __init__(self, in_channels=27, group_out_channels=8, final_out_channels=64):
        super(GroupConv3DRelu, self).__init__()
        
        self.pre_conv3d = nn.Conv3d(in_channels=3, out_channels=3, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.group_out_channels = group_out_channels
        self.relu = nn.ReLU(inplace=True)
        # 定义每组的卷积层
        self.conv1 = nn.Conv2d(7, group_out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(1, group_out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(1, group_out_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(7, group_out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(1, group_out_channels, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(1, group_out_channels, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(7, group_out_channels, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(1, group_out_channels, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(1, group_out_channels, kernel_size=3, padding=1)

        self.bn = nn.BatchNorm2d(group_out_channels)

        self.bn3d = nn.BatchNorm3d(3)

        self.post_conv3d = nn.Conv3d(in_channels=3, out_channels=3, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        
        # 汇总卷积层
        self.final_conv = nn.Conv2d(group_out_channels * 9, final_out_channels, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)


    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, 3, 9, height, width)
        x = self.pre_conv3d(x)
        x = self.relu(self.bn3d(x))

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
        out1 = self.relu(self.conv1(group1))
        out2 = self.relu(self.conv2(group2))
        out3 = self.relu(self.conv3(group3))
        out4 = self.relu(self.conv4(group4))
        out5 = self.relu(self.conv5(group5))
        out6 = self.relu(self.conv6(group6))
        out7 = self.relu(self.conv7(group7))
        out8 = self.relu(self.conv8(group8))
        out9 = self.relu(self.conv9(group9))
        
        concatenated = torch.cat([out1, out2, out3, out4, out5, out6, out7, out8, out9], dim=1)
        
        # print(f'shape concatenated {concatenated.shape}')
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, 3, -1, height, width)
        x = self.post_conv3d(x)
        x = self.relu(self.bn3d(x))
        x = x.view(batch_size, -1, height, width)

        final_out = self.final_conv(concatenated)
        
        return final_out

class GroupConv3D(nn.Module):
    def __init__(self, in_channels=27, group_out_channels=8, final_out_channels=64):
        super(GroupConv3D, self).__init__()
        
        self.pre_conv3d = nn.Conv3d(in_channels=3, out_channels=3, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.group_out_channels = group_out_channels
        self.relu = nn.ReLU(inplace=True)
        # 定义每组的卷积层
        self.conv1 = nn.Conv2d(7, group_out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(1, group_out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(1, group_out_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(7, group_out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(1, group_out_channels, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(1, group_out_channels, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(7, group_out_channels, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(1, group_out_channels, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(1, group_out_channels, kernel_size=3, padding=1)

        self.post_conv3d = nn.Conv3d(in_channels=3, out_channels=3, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        
        # 汇总卷积层
        self.final_conv = nn.Conv2d(group_out_channels * 9, final_out_channels, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

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

# 定义 ResNet18
class GroupResNet(nn.Module):
    def __init__(self, input_channel, block, layers, num_classes=1000):
        super(GroupResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = GroupConv()
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 每个阶段的残差块
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class GroupResNet3D(nn.Module):
    def __init__(self, input_channel, block, layers, num_classes=1000):
        super(GroupResNet3D, self).__init__()
        self.in_channels = 64
        self.conv1 = GroupConv3D()
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 每个阶段的残差块
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def extract_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def forward(self, x):
        x = self.extract_features(x)
        x = self.fc(x)

        return x

class GroupResNet3DRelu(nn.Module):
    def __init__(self, input_channel, block, layers, num_classes=1000):
        super(GroupResNet3DRelu, self).__init__()
        self.in_channels = 64
        self.conv1 = GroupConv3DRelu()
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 每个阶段的残差块
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
# 实例化 ResNet18
def groupResnet18(num_classes=1000, input_channel=10):
    model = GroupResNet(input_channel, BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    return model

def groupResnet3D18(num_classes=1000, input_channel=10):
    model = GroupResNet3D(input_channel, BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    return model

def groupResnet3D18Relu(num_classes=1000, input_channel=10):
    model = GroupResNet3DRelu(input_channel, BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    return model

def groupResnet3D34(num_classes=1000, input_channel=10):
    model = GroupResNet3D(input_channel, BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    return model

def groupFCA18(num_classes=1000, input_channel=10, pretrained=False):
    """Constructs a FcaNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = GroupFCAResNet(input_channel, FcaBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    return model

def groupFCA3D18(num_classes=1000, input_channel=10, pretrained=False):
    """Constructs a FcaNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = GroupFCAResNet3D(input_channel, FcaBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    return model