import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np


def get_label(bs, device):
    label_r = torch.full((bs, ), 1, device=device)
    label_f = torch.full((bs, ), 0, device=device)
    return label_r, label_f


class SGAN(nn.Module):
    def __init__(self, device):
        super(SGAN, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.device = device

    def d_loss(self, c_xr, c_xf):
        bs = c_xf.shape[0]
        label_r, label_f = get_label(bs, self.device)
        return self.criterion(c_xr, label_r) + self.criterion(c_xf, label_f)

    def g_loss(self, c_xf):
        bs = c_xf.shape[0]
        label_r, _ = get_label(bs, self.device)
        return self.criterion(c_xf, label_r)


class LSGAN(nn.Module):
    def __init__(self, device):
        super(LSGAN, self).__init__()
        self.device = device

    def d_loss(self, c_xr, c_xf):
        bs = c_xf.shape[0]
        label_r, label_f = get_label(bs, self.device)
        v1, v2 = (c_xr - label_r) ** 2, (c_xf - label_f) ** 2
        return torch.mean(v1) + torch.mean(v2)

    def g_loss(self, c_xf):
        bs = c_xf.shape[0]
        label_r, _ = get_label(bs, self.device)
        return torch.mean((c_xf - label_r) ** 2)


class WGANGP(nn.Module):
    def __init__(self, device):
        super(WGANGP, self).__init__()
        self.device = device

    def d_loss(self, c_xr, c_xf):
        return -torch.mean(c_xr) + torch.mean(c_xf)

    def g_loss(self, c_xf):
        return -torch.mean(c_xf)

    def gradient_penalty(self, netD, real_image, fake_image):
        bs = real_image.shape[0]
        device = real_image.device
        alpha = torch.FloatTensor(bs, 1, 1, 1).uniform_(
            0, 1).expand(real_image.size()).to(device)
        interpolation = alpha * real_image + (1 - alpha) * fake_image

        c_xi = netD(interpolation)
        grad = autograd.grad(c_xi, interpolation, torch.ones(c_xi.size()).to(
                             device), create_graph=True, retain_graph=True)[0]
        grad = grad.view(bs, -1)
        penalty = torch.mean((grad.norm(2, dim=1) - 1) ** 2)
        return penalty


class NonSaturatingR1(nn.Module):
    def __init__(self, device):
        super(NonSaturatingR1, self).__init__()
        self.device = device

    def d_loss(self, c_xr, c_xf):
        return torch.mean(F.softplus(-c_xr)) + torch.mean(F.softplus(c_xf))

    def g_loss(self, c_xf):
        return torch.mean(F.softplus(-c_xf))

    def gradient_penalty(self, netD, real_image, fake_image):
        bs = real_image.shape[0]
        device = real_image.device

        real_image.requires_grad = True
        c_xi = netD(real_image)
        grad = autograd.grad(c_xi, real_image, torch.ones(c_xi.size()).to(
                             device), create_graph=True, retain_graph=True)[0]
        grad = grad.view(bs, -1)
        penalty = torch.mean(grad.norm(2, dim=1) ** 2) / 2.0
        return penalty


class HINGEGAN(nn.Module):
    def __init__(self, device):
        super(HINGEGAN, self).__init__()
        self.device = device

    def d_loss(self, c_xr, c_xf):
        bs = c_xf.shape[0]
        v1, v2 = torch.nn.ReLU()(1 - c_xr), torch.nn.ReLU()(1 + c_xf)
        return torch.mean(v1) + torch.mean(v2)

    def g_loss(self, c_xf):
        return -torch.mean(c_xf)
