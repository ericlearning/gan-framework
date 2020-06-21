import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm
from ..modules import sBN, Reshape


class DCGANGenerator64(nn.Module):
    def __init__(self, nz=128):
        super(DCGANGenerator64, self).__init__()

        self.deconv0 = nn.Sequential(
            Reshape(-1, nz, 1, 1),
            nn.ConvTranspose2d(nz, 512, 4, 1, 0)
        )
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv1 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1)
        )
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv2 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1)
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv3 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1)
        )
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv4 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

        for m in self.modules():
            if(isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d))):
                m.weight.data.normal_(0.0, 0.02)
                if(m.bias is not None):
                    m.bias.data.zero_()

    def forward(self, z):
        out = self.deconv0(z)
        out = self.deconv1(self.bn1(out))
        out = self.deconv2(self.bn2(out))
        out = self.deconv3(self.bn3(out))
        out = self.deconv4(self.bn4(out))
        return out


class DCGANDiscriminator64(nn.Module):
    def __init__(self):
        super(DCGANDiscriminator64, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, 4, 2, 1)
        )
        self.bn1 = nn.BatchNorm2d(128)
        self.conv1 = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, 4, 2, 1)
        )
        self.bn2 = nn.BatchNorm2d(256)
        self.conv2 = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 4, 2, 1)
        )
        self.bn3 = nn.BatchNorm2d(512)
        self.conv3 = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1, 4, 1, 0)
        )
        for m in self.modules():
            if(isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d))):
                m.weight.data.normal_(0.0, 0.02)
                if(m.bias is not None):
                    m.bias.data.zero_()

    def forward(self, z):
        out = self.conv0(z)
        out = self.conv1(self.bn1(out))
        out = self.conv2(self.bn2(out))
        out = self.conv3(self.bn3(out))
        return out
