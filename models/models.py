import torch.nn as nn


__all__ = ['AlexNet', 'MLP', 'CNN2', 'CNN3']


def AlexNet(n_channels):  # from li paper
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


def MLP(n_channels):  # from li paper
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


def CNN2(n_channels):  # from banerjee paper
    act = nn.ReLU
    return nn.Sequential(
        nn.Conv2d(n_channels, 32, kernel_size=5),
        act(),
        nn.MaxPool2d(kernel_size=2),
        act(),
        nn.Conv2d(32, 64, kernel_size=5),
        act(),
        nn.MaxPool2d(kernel_size=2),
        act(),
        nn.Flatten(),
        nn.LazyLinear(1024),
        act(),
        nn.Linear(1024, 10)
    )


def CNN3(n_channels):  # from banerjee paper
    act = nn.ReLU
    return nn.Sequential(
        nn.Conv2d(n_channels, 64, kernel_size=5),
        act(),
        nn.MaxPool2d(kernel_size=2),
        act(),
        nn.Conv2d(64, 192, kernel_size=5),
        act(),
        nn.MaxPool2d(kernel_size=2),
        act(),
        nn.Flatten(),
        nn.LazyLinear(384),
        act(),
        nn.Linear(384, 192),
        act(),
        nn.Linear(192, 10)
    )

