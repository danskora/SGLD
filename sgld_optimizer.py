import torch
from torch.distributions import Normal
from torch.optim import Optimizer
import numpy as np


class SGLD(Optimizer):
    def __init__(self, params, lr):
        super(SGLD, self).__init__(params, dict(lr=lr))

    def step(self, closure=None):
        loss = None

        for group in self.param_groups:

            for p in group['params']:

                if p.grad is None:
                    continue

                gradient = p.grad.data
                size = gradient.size()
                noise = Normal(torch.zeros(size), torch.ones(size) * np.sqrt(group['lr']))
                gradient_w_noise = gradient + noise.sample()
                p.data.add_(gradient_w_noise, alpha=-group['lr'])

        return loss


# new optimizer to suit the noise variance of the MNIST experiment
class NewSGLD(Optimizer):
    def __init__(self, params, lr, variance):
        super(NewSGLD, self).__init__(params, dict(lr=lr))
        self.variance = variance

    def step(self, closure=None):
        loss = None

        for group in self.param_groups:

            for p in group['params']:

                if p.grad is None:
                    continue

                gradient = p.grad.data  # is this correct
                size = gradient.size()
                noise = Normal(torch.zeros(size), torch.ones(size) * self.variance)
                gradient_w_noise = gradient + noise.sample()
                p.data.add_(gradient_w_noise, alpha=-group['lr'])

        return loss
