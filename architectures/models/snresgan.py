import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm
from ..modules import sBN, Reshape, Upsample


class ResBlockDisFirst(nn.Module):
    def __init__(self, ni, no):
        super(ResBlockDisFirst, self).__init__()
        self.conv1 = SpectralNorm(nn.Conv2d(ni, no, 3, 1, 1))
        self.conv2 = SpectralNorm(nn.Conv2d(no, no, 3, 1, 1))
        self.conv_s = SpectralNorm(nn.Conv2d(ni, no, 1, 1, 0))
        self.relu = nn.ReLU(True)

        nn.init.xavier_uniform_(self.conv1.weight.data, math.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, math.sqrt(2))
        nn.init.xavier_uniform_(self.conv_s.weight.data, 1.0)

    def branch(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = F.avg_pool2d(out, 2)
        return out

    def shortcut(self, x):
        out = F.avg_pool2d(x, 2)
        out = self.conv_s(out)
        return out

    def forward(self, x):
        out = self.branch(x) + self.shortcut(x)
        return out


class ResBlockDis(nn.Module):
    def __init__(self, ni, no, nh=None, downsample=True):
        super(ResBlockDis, self).__init__()
        nh = ni if nh is None else nh
        self.downsample = downsample
        self.conv1 = SpectralNorm(nn.Conv2d(ni, nh, 3, 1, 1))
        self.conv2 = SpectralNorm(nn.Conv2d(nh, no, 3, 1, 1))

        self.shortcut_con = self.downsample or ni != no
        if self.shortcut_con:
            self.conv_s = SpectralNorm(nn.Conv2d(ni, no, 1, 1, 0))
        self.relu = nn.ReLU(True)

        nn.init.xavier_uniform_(self.conv1.weight.data, math.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, math.sqrt(2))
        if self.shortcut_con:
            nn.init.xavier_uniform_(self.conv_s.weight.data, 1.0)

    def branch(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample:
            out = F.avg_pool2d(out, 2)
        return out

    def shortcut(self, x):
        out = x
        if self.shortcut_con:
            out = self.conv_s(out)
            if self.downsample:
                out = F.avg_pool2d(out, 2)
        return out

    def forward(self, x):
        out = self.branch(x) + self.shortcut(x)
        return out


class ResBlockGen(nn.Module):
    def __init__(self, ni, no, nh=None, upsample=True):
        super(ResBlockGen, self).__init__()
        nh = ni if nh is None else nh
        self.upsample = upsample
        self.conv1 = nn.Conv2d(ni, nh, 3, 1, 1)
        self.conv2 = nn.Conv2d(nh, no, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(ni)
        self.bn2 = nn.BatchNorm2d(nh)

        self.shortcut_con = self.upsample or ni != no
        if self.shortcut_con:
            self.conv_s = nn.Conv2d(ni, no, 1, 1, 0)
        self.relu = nn.ReLU(True)

        nn.init.xavier_uniform_(self.conv1.weight.data, math.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, math.sqrt(2))
        if self.shortcut_con:
            nn.init.xavier_uniform_(self.conv_s.weight.data, 1.0)

    def branch(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        if self.upsample:
            out = F.interpolate(out, None, 2, 'bilinear',
                                align_corners=False)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out

    def shortcut(self, x):
        out = x
        if self.shortcut_con:
            if self.upsample:
                out = F.interpolate(out, None, 2, 'bilinear',
                                    align_corners=False)
            out = self.conv_s(out)
        return out

    def forward(self, x):
        out = self.branch(x) + self.shortcut(x)
        return out


class SNResNetGenerator128(nn.Module):
    def __init__(self, nz=128):
        super(SNResNetGenerator128, self).__init__()
        self.fc = nn.Linear(nz, 1024 * 4 * 4)
        self.blks = nn.Sequential(
            ResBlockGen(1024, 1024),
            ResBlockGen(1024, 512),
            ResBlockGen(512, 256),
            ResBlockGen(256, 128),
            ResBlockGen(128, 64),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv = nn.Conv2d(64, 3, 3, 1, 1)
        self.tanh = nn.Tanh()

        nn.init.xavier_uniform_(self.fc.weight.data, 1.0)
        nn.init.xavier_uniform_(self.conv.weight.data, 1.0)

    def forward(self, z):
        out = self.fc(z).reshape(-1, 1024, 4, 4)
        out = self.blks(out)
        out = self.conv(out)
        out = self.tanh(out)
        return out


class SNResNetDiscriminator128(nn.Module):
    def __init__(self):
        super(SNResNetDiscriminator128, self).__init__()
        self.blks = nn.Sequential(
            ResBlockDisFirst(3, 64),
            ResBlockDis(64, 128),
            ResBlockDis(128, 256),
            ResBlockDis(256, 512),
            ResBlockDis(512, 1024),
            ResBlockDis(1024, 1024, downsample=False),
            nn.ReLU()
        )
        self.fc = SpectralNorm(nn.Linear(1024, 1))

        nn.init.xavier_uniform_(self.fc.weight.data, 1.0)

    def forward(self, z):
        out = self.blks(z)
        out = torch.sum(out, (2, 3))
        out = self.fc(out)
        return out


class SNResNetGenerator64(nn.Module):
    def __init__(self, nz=128):
        super(SNResNetGenerator64, self).__init__()
        self.fc = nn.Linear(nz, 1024 * 4 * 4)
        self.blks = nn.Sequential(
            ResBlockGen(1024, 512),
            ResBlockGen(512, 256),
            ResBlockGen(256, 128),
            ResBlockGen(128, 64),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv = nn.Conv2d(64, 3, 3, 1, 1)
        self.tanh = nn.Tanh()

        nn.init.xavier_uniform_(self.fc.weight.data, 1.0)
        nn.init.xavier_uniform_(self.conv.weight.data, 1.0)

    def forward(self, z):
        out = self.fc(z).reshape(-1, 1024, 4, 4)
        out = self.blks(out)
        out = self.conv(out)
        out = self.tanh(out)
        return out


class SNResNetDiscriminator64(nn.Module):
    def __init__(self):
        super(SNResNetDiscriminator64, self).__init__()
        self.blks = nn.Sequential(
            ResBlockDisFirst(3, 64),
            ResBlockDis(64, 128),
            ResBlockDis(128, 256),
            ResBlockDis(256, 512),
            ResBlockDis(512, 1024),
            nn.ReLU()
        )
        self.fc = SpectralNorm(nn.Linear(1024, 1))

        nn.init.xavier_uniform_(self.fc.weight.data, 1.0)

    def forward(self, z):
        out = self.blks(z)
        out = torch.sum(out, (2, 3))
        out = self.fc(out)
        return out
