import torch.nn as nn


__all__ = ['make_alexnet', 'make_mlp']


def make_alexnet(n_channels):
    act = nn.ReLU
    return nn.Sequential(
        nn.Conv2d(n_channels, 64, kernel_size=5),
        act(),
        nn.MaxPool2d(kernel_size=3),
        act(),
        nn.Conv2d(64, 192, kernel_size=5),
        act(),
        nn.MaxPool2d(kernel_size=3),
        act(),
        nn.Flatten(),
        nn.LazyLinear(384),
        act(),
        nn.Linear(384, 192),
        act(),
        nn.Linear(192, 10)
    )


def make_mlp():
    act = nn.ReLU
    return nn.Sequential(
        nn.Flatten(),
        nn.LazyLinear(512),
        act(),
        nn.Linear(512, 512),
        act(),
        nn.Linear(512, 512),
        act(),
        nn.Linear(512, 10)
    )

