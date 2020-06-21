import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm
from ..modules import sBN, Reshape, Upsample, AdaIN, LSTMCell


class AdaINDiscriminator64(nn.Module):
    def __init__(self):
        super(AdaINDiscriminator64, self).__init__()
        self.blk1 = INConvBlock(3, 32)
        self.blk2 = INConvBlock(32, 64)
        self.blk3 = INConvBlock(64, 128)
        self.blk4 = INConvBlock(128, 256)
        self.blk5 = INConvBlock(256, 512)
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


class AdaINGenerator64(nn.Module):
    def __init__(self, nz=128):
        super(AdaINGenerator64, self).__init__()
        self.start = nn.Parameter(torch.randn(1, 512, 4, 4))
        self.blk1 = AdaINConvBlock_T1(512, 256, nz, True)
        self.blk2 = AdaINConvBlock_T1(256, 128, nz, True)
        self.blk3 = AdaINConvBlock_T1(128, 64, nz, True)
        self.blk4 = AdaINConvBlock_T1(64, 64, nz, False)
        self.blk5 = AdaINConvBlock_T1(64, 32, nz, True)
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
        out = self.conv(out)
        out = self.tanh(out)
        return out


class AdaINForeGenerator64(nn.Module):
    def __init__(self, nz=128):
        super(AdaINGenerator64, self).__init__()
        self.start = nn.Parameter(torch.randn(1, 512, 4, 4))
        self.fg_conv = nn.Conv2d(32, 3, 3, 1, 1)
        self.mask_conv = nn.Conv2d(32, 1, 3, 1, 1)
        self.common = nn.Sequential([
            AdaINConvBlock_T1(512, 256, nz, True),
            AdaINConvBlock_T1(256, 128, nz, True),
            AdaINConvBlock_T1(128, 64, nz, True),
            AdaINConvBlock_T1(64, 64, nz, False)
        ])
        self.fg = nn.Sequential([
            AdaINConvBlock_T1(64, 32, nz, True),
            self.fore_conv,
            nn.Tanh()
        ])
        self.mask = nn.Sequential([
            AdaINConvBlock_T1(64, 32, nz, True),
            self.mask_conv,
            nn.Tanh()
        ])
        nn.init.xavier_uniform_(self.fg_conv.weight.data, 1.0)
        nn.init.xavier_uniform_(self.mask_conv.weight.data, 1.0)

    def forward(self, z):
        out = self.start
        out = self.common(out)
        fg = self.fg(out)
        mask = (self.mask(out) + 1) / 2
        return fg, mask


class NaiveHAGANGenerator64(nn.Module):
    def __init__(self, nz=128):
        super(NaiveHAGANGenerator64, self).__init__()
        self.nz = nz
        self.cell = LSTMCell(nz, nz)
        self.bg_gen = AdaINGenerator64(nz)
        self.fg_gen = AdaINForeGenerator64(nz)

    def composer(self, bg, fgs, masks):
        img = bg.clone()
        for fg, mask in zip(fgs, masks):
            img = img * (1 - mask) + fg * mask
        return img

    def forward(self, x, hiddens=None):
        # x: (object_num, bs, nz)
        object_num, bs, _ = x.shape

        # h, c: (bs, nz)
        if hidden is None:
            h = torch.zeros(bs, self.nz).to(x.device)
            c = torch.zeros(bs, self.nz).to(x.device)

        # generate background with x[0]
        bg = self.bg_gen(x[0])

        fgs, masks = [], []
        for i in range(object_num):
            x_i = x[i]
            h, c = self.cell(x_i, h, c)

            # generate foreground with h
            fg, mask = self.fg_gen(h)
            fgs.append(fg)
            masks.append(mask)

        out = self.composer(bg, fgs, masks)
        return out, bg, fgs, masks
