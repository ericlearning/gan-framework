import os
import cv2
import torch
import numpy as np
from losses import *
from torchvision import transforms


def get_display_samples(samples, num_samples_y, num_samples_x):
    sz = samples[0].shape[0]
    nc = samples[0].shape[2]
    display = np.zeros((sz * num_samples_y, sz * num_samples_x, nc))
    for i in range(num_samples_y):
        for j in range(num_samples_x):
            y_st, y_en = i * sz, (i + 1) * sz
            x_st, x_en = j * sz, (j + 1) * sz
            display[y_st:y_en, x_st:x_en, :] = cv2.cvtColor(
                samples[i * num_samples_x + j] * 255.0, cv2.COLOR_BGR2RGB)
    return display.astype(np.uint8)


def save(filename, netD, netG, optD, optG):
    state = {
        'netD': netD.state_dict(),
        'netG': netG.state_dict(),
        'optD': optD.state_dict(),
        'optG': optG.state_dict()
    }
    torch.save(state, filename)


def load(filename, netD, netG, optD, optG):
    state = torch.load(filename)
    netD.load_state_dict(state['netD'])
    netG.load_state_dict(state['netG'])
    optD.load_state_dict(state['optD'])
    optG.load_state_dict(state['optG'])


def save_fig(filename, fig):
    fig.savefig(filename)


def get_sample_images_list(fixed_z, netG):
    netG.eval()
    with torch.no_grad():
        fake = netG(fixed_z).detach().cpu().numpy()
        sample_images_list = []
        for j in range(49):
            cur_img = (fake[j] + 1) / 2.0
            sample_images_list.append(cur_img.transpose(1, 2, 0))

    netG.train()

    return sample_images_list


def get_gan_loss(device, loss_type):
    loss_dict = {'SGAN': SGAN, 'LSGAN': LSGAN, 'HINGEGAN': HINGEGAN,
                 'WGANGP': WGANGP, 'R1': NonSaturatingR1}
    loss = loss_dict[loss_type](device)

    return loss


def get_default_dt(basic_types, sz):
    if(basic_types is None):
        dt = transforms.Compose([
            transforms.Resize(sz),
            transforms.CenterCrop(sz),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    elif(basic_types == 'MNIST'):
        dt = transforms.Compose([
            transforms.Resize(sz),
            transforms.CenterCrop(sz),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    elif(basic_types == 'CIFAR10'):
        dt = transforms.Compose([
            transforms.Resize(sz),
            transforms.CenterCrop(sz),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    elif(basic_types == 'CelebA'):
        dt = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(sz),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    return dt
