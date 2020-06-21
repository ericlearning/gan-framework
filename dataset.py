import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils import get_default_dt


class Dataset():
    def __init__(self, train_dir, basic_types=None, shuffle=True):
        self.train_dir = train_dir
        self.basic_types = basic_types
        self.shuffle = shuffle

    def get_loader(self, sz, bs, dt=None, num_workers=10):
        if(dt is None):
            dt = get_default_dt(self.basic_types, sz)

        if(self.basic_types is None):
            train_dataset = datasets.ImageFolder(self.train_dir, dt)
        elif(self.basic_types == 'MNIST'):
            train_dataset = datasets.MNIST(
                self.train_dir, train=True, download=True, transform=dt)
        elif(self.basic_types == 'CIFAR10'):
            train_dataset = datasets.CIFAR10(
                self.train_dir, train=True, download=True, transform=dt)
        elif(self.basic_types == 'CelebA'):
            train_dataset = datasets.CelebA(
                self.train_dir, download=True, transform=dt)

        train_loader = DataLoader(
            train_dataset, batch_size=bs, shuffle=self.shuffle,
            num_workers=num_workers)

        return train_loader
