import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm
from ..modules import sBN, Reshape


class SelfModGenerator(nn.Module):
    def __init__(self, sz, nz=128):
        super(SelfModGenerator, self).__init__()
        init_sz = sz // 16
        self.fc = nn.Sequential(
            SpectralNorm(nn.Linear(nz, 512 * init_sz * init_sz)),
            Reshape(-1, 512, init_sz, init_sz)
        )
        self.bn1 = sBN(512, nz)
        self.deconv1 = nn.Sequential(
            nn.ReLU(),
            SpectralNorm(nn.ConvTranspose2d(512, 256, 4, 2, 1))
        )
        self.bn2 = sBN(256, nz)
        self.deconv2 = nn.Sequential(
            nn.ReLU(),
            SpectralNorm(nn.ConvTranspose2d(256, 128, 4, 2, 1))
        )
        self.bn3 = sBN(128, nz)
        self.deconv3 = nn.Sequential(
            nn.ReLU(),
            SpectralNorm(nn.ConvTranspose2d(128, 64, 4, 2, 1))
        )
        self.bn4 = sBN(64, nz)
        self.deconv4 = nn.Sequential(
            nn.ReLU(),
            SpectralNorm(nn.ConvTranspose2d(64, 3, 4, 2, 1)),
            nn.Tanh()
        )
        for m in self.modules():
            if(isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d))):
                m.weight.data.normal_(0.0, 0.02)
                if(m.bias is not None):
                    m.bias.data.zero_()

    def forward(self, z):
        out = self.fc(z)
        out = self.deconv1(self.bn1(out, z))
        out = self.deconv2(self.bn2(out, z))
        out = self.deconv3(self.bn3(out, z))
        out = self.deconv4(self.bn4(out, z))
        return out


class SelfModDiscriminator(nn.Module):
    def __init__(self, sz):
        super(SelfModDiscriminator, self).__init__()
        final_sz = sz // 8
        self.conv1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(3, 64, 3, 1, 1)),
            nn.LeakyReLU(0.1)
        )
        self.conv2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.1)
        )
        self.conv3 = nn.Sequential(
            SpectralNorm(nn.Conv2d(128, 128, 3, 1, 1)),
            nn.LeakyReLU(0.1)
        )
        self.conv4 = nn.Sequential(
            SpectralNorm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.LeakyReLU(0.1)
        )
        self.conv5 = nn.Sequential(
            SpectralNorm(nn.Conv2d(256, 256, 3, 1, 1)),
            nn.LeakyReLU(0.1)
        )
        self.conv6 = nn.Sequential(
            SpectralNorm(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.LeakyReLU(0.1)
        )
        self.conv7 = nn.Sequential(
            SpectralNorm(nn.Conv2d(512, 512, 3, 1, 1)),
            nn.LeakyReLU(0.1)
        )
        self.fc = nn.Sequential(
            Reshape(-1, 512 * final_sz * final_sz),
            SpectralNorm(nn.Linear(512 * final_sz * final_sz, 1))
        )
        for m in self.modules():
            if(isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d))):
                m.weight.data.normal_(0.0, 0.02)
                if(m.bias is not None):
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.fc(out)
        return out
