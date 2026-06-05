import torch
import torch.nn as nn
import torch.fft as fft

class FourierTransformBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(FourierTransformBlock, self).__init__()
        self.stride = stride
        # Translated comment
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_real = nn.BatchNorm2d(out_channels)
        self.bn_imag = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Apply FFT to the input tensor
        x_fft = fft.fftn(x, dim=(-2, -1))
        
        # Extract real and imaginary parts
        real = x_fft.real
        imag = x_fft.imag

        # Apply convolution to real and imaginary parts
        real = self.conv_real(real)
        real = self.bn_real(real)
        
        imag = self.conv_imag(imag)
        imag = self.bn_imag(imag)

        # Reconstruct the complex tensor
        x_fft_processed = torch.complex(real, imag)
        
        # Apply inverse FFT to bring back to spatial domain
        x_ifft = fft.ifftn(x_fft_processed, dim=(-2, -1))
        
        # Return the real part as the output
        return x_ifft.real

# Basic Block with Parallel Conv and Fourier Branches
class BasicBlockWithParallelBranches(nn.Module):
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlockWithParallelBranches, self).__init__()
        # Convolutional branch
        self.conv_branch = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )
        
        # Fourier branch, ensuring the same stride
        self.fourier_branch = FourierTransformBlock(in_planes, planes, stride=stride)

        # Downsample layer (for identity connection)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        # Convolutional branch
        conv_out = self.conv_branch(x)
        
        # Fourier branch
        fourier_out = self.fourier_branch(x)
        
        # Combine the outputs from both branches
        out = conv_out + fourier_out

        # Downsample if needed
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add identity connection
        out += identity
        out = self.relu(out)

        return out

# ResNet with Parallel Conv and Fourier Branches
class FreResNet(nn.Module):
    def __init__(self, block, layers, input_channel = 18, num_classes=1000):
        super(FreResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

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

def frequency_resnet18(input_channel = 18, num_classes=1000):
    return FreResNet(BasicBlockWithParallelBranches, [2, 2, 2, 2], input_channel, num_classes)