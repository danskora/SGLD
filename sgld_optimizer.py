import torch
from torch.distributions import Normal
from torch.optim import Optimizer
import numpy as np


class SGLD(Optimizer):
    def __init__(self, params, lr=0.1):
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
