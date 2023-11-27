import random  # seed random somewhere?  and how does torch seed work?  what is its domain?

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Subset


class NoisyMNIST(torchvision.datasets.MNIST):
    def __init__(self, noise, size=None, train=True):
        if not size:
            size = 60000 if train else 10000
        super(NoisyMNIST, self).__init__(root='./data', train=train, download=True,
                                         transform=transforms.ToTensor())

        self.newlabels = {}
        indices = torch.randperm(size)[:int(noise*size)].tolist()
        for idx in indices:
            self.newlabels[idx] = int(random.random()*10)

    def __getitem__(self, idx):
        img, label = super(NoisyMNIST, self).__getitem__(idx)
        return img, self.newlabels.get(idx, label)


class NoisyCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, noise, size=None, train=True):
        if not size:
            size = 50000 if train else 10000
        super(NoisyCIFAR10, self).__init__(root='./data', train=train, download=True,
                                           transform=transforms.ToTensor())

        self.newlabels = {}
        indices = torch.randperm(size)[:int(noise*size)].tolist()
        for idx in indices:
            self.newlabels[idx] = int(random.random()*10)

    def __getitem__(self, idx):
        img, label = super(NoisyCIFAR10, self).__getitem__(idx)
        return img, self.newlabels.get(idx, label)


def get_dataset(dataset, noise=0.0, size=None, train=True):
    if dataset == 'MNIST':
        d = NoisyMNIST(noise, size=size, train=train)
    elif dataset == 'CIFAR10':
        d = NoisyCIFAR10(noise, size=size, train=train)
    else:
        raise NotImplementedError

    if size:
        return Subset(d, range(size))
    else:
        return d

