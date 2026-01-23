import torch
import torch.nn as nn
import numpy as np

class FFTLayer(nn.Module):
    def __init__(self):
        super(FFTLayer, self).__init__()

    def forward(self, x):
        batch, channels, height, width = x.shape
        x_fft = torch.fft.fft2(x, norm='ortho')  # Add norm='ortho' for normalized FFT

        # Extract magnitude and phase for each channel
        magnitude = torch.log(torch.abs(x_fft) + 1e-8)
        phase = torch.angle(x_fft)

        # Stack magnitude and phase along channel dimension
        # Output will have twice the number of channels: 4 channels -> 8 channels
        return torch.cat((magnitude, phase), dim=1)