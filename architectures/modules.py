import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm


class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x):
        return F.interpolate(x, None, 2, 'bilinear',
                             align_corners=False)


class Downsample(nn.Module):
    def __init__(self):
        super(Downsample, self).__init__()

    def forward(self, x):
        return F.interpolate(x, None, 0.5, 'bilinear',
                             align_corners=False)


class sBN(nn.Module):
    def __init__(self, nf, nz, nh=32):
        super(sBN, self).__init__()
        self.nf = nf
        self.bn = nn.BatchNorm2d(nf, affine=False)
        self.gamma = nn.Sequential(
            nn.Linear(nz, nh),
            nn.ReLU(),
            nn.Linear(nh, nf, bias=False),
            Reshape(-1, nf, 1, 1)
        )
        self.beta = nn.Sequential(
            nn.Linear(nz, nh),
            nn.ReLU(),
            nn.Linear(nh, nf, bias=False),
            Reshape(-1, nf, 1, 1)
        )

    def forward(self, x, z):
        return self.bn(x) * self.gamma(z) + self.beta(z)


class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        out = x.view(*self.shape)
        return out


class AdaIN(nn.Module):
    def __init__(self, ni, nz):
        super(AdaIN, self).__init__()
        self.ni = ni
        self.norm = nn.InstanceNorm2d(ni, affine=False)
        self.fc = nn.Linear(nz, ni * 2)
        nn.init.xavier_uniform_(self.fc.weight.data, 1.0)

    def forward(self, x, z):
        gamma, beta = self.fc(z).split(self.ni, 1)
        gamma = gamma.reshape(-1, self.ni, 1, 1)
        beta = beta.reshape(-1, self.ni, 1, 1)
        out = self.norm(x) * gamma + beta
        return out


class LSTMCell(nn.Module):
    def __init__(self, ic, hc):
        super(LSTMCell, self).__init__()
        self.ic, self.hc = ic, hc
        self.linear_i = nn.Linear(ic, hc * 4)
        self.linear_h = nn.Linear(hc, hc * 4)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

    def forward(self, x, h, c):
        out_i, out_h = self.linear_i(x), self.linear_h(h)
        out_i, out_h = out_i.split(self.hc, 1), out_h.split(self.hc, 1)

        forget_gate = self.sig(out_i[0] + out_h[0])
        input_gate = self.sig(out_i[1] + out_h[1])
        output_gate = self.sig(out_i[2] + out_h[2])
        modulation_gate = self.tanh(out_i[3] + out_h[3])

        c = c * forget_gate + modulation_gate * input_gate
        h = output_gate * self.tanh(c)
        return h, c
