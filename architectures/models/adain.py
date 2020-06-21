import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm
from ..modules import sBN, Reshape, Upsample, AdaIN


class AdaINConvBlock_T1(nn.Module):
    def __init__(self, ni, no, nz, upsample=True):
        super(AdaINConvBlock_T1, self).__init__()
        self.upsample = upsample
        self.conv = nn.Conv2d(ni, no, 3, 1, 1)
        self.adain = AdaIN(ni, nz)
        self.lrelu = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.conv.weight.data, 1.0)

    def forward(self, x, z):
        out = x
        out = self.adain(out, z)
        if self.upsample:
            out = F.interpolate(out, None, 2, 'bilinear',
                                align_corners=False)
        out = self.conv(out)
        out = self.lrelu(out)

        return out


class ConvBlock(nn.Module):
    def __init__(self, ni, no):
        super(ConvBlock, self).__init__()
        self.conv = SpectralNorm(nn.Conv2d(ni, no, 3, 2, 1))
        self.lrelu = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.conv.weight.data, 1.0)

    def forward(self, x):
        out = self.conv(x)
        out = self.lrelu(out)
        return out


class AdaINGenerator64(nn.Module):
    def __init__(self, nz=128):
        super(AdaINGenerator64, self).__init__()
        self.start = nn.Parameter(torch.randn(1, 512, 4, 4))
        self.blk1 = AdaINConvBlock_T1(512, 256, nz, True)
        self.blk2 = AdaINConvBlock_T1(256, 128, nz, True)
        self.blk3 = AdaINConvBlock_T1(128, 128, nz, False)
        self.blk4 = AdaINConvBlock_T1(128, 128, nz, False)
        self.blk5 = AdaINConvBlock_T1(128, 64, nz, True)
        self.blk6 = AdaINConvBlock_T1(64, 32, nz, True)
        self.conv = nn.Conv2d(32, 3, 3, 1, 1)
        self.tanh = nn.Tanh()
        nn.init.xavier_uniform_(self.conv.weight.data, 1.0)

    def forward(self, z):
        out = self.start
        out = self.blk1(out, z)
        out = self.blk2(out, z)
        out = self.blk3(out, z)
        out = self.blk4(out, z)
        out = self.blk5(out, z)
        out = self.blk6(out, z)
        out = self.conv(out)
        out = self.tanh(out)
        return out


class AdaINDiscriminator64(nn.Module):
    def __init__(self):
        super(AdaINDiscriminator64, self).__init__()
        self.blk1 = ConvBlock(3, 32)
        self.blk2 = ConvBlock(32, 64)
        self.blk3 = ConvBlock(64, 128)
        self.blk4 = ConvBlock(128, 256)
        self.blk5 = ConvBlock(256, 512)
        self.fc = SpectralNorm(nn.Linear(512, 1))
        nn.init.xavier_uniform_(self.fc.weight.data, 1.0)

    def forward(self, x):
        out = self.blk1(x)
        out = self.blk2(out)
        out = self.blk3(out)
        out = self.blk4(out)
        out = self.blk5(out)
        out = torch.sum(out, (2, 3))
        out = self.fc(out)
        return out
