import torch
from torch.distributions import Normal
from torch.optim import Optimizer
import math


# class SGLD(Optimizer):  # check if this even reflects the original paper or just that blog post
#     def __init__(self, params, lr):
#         super(SGLD, self).__init__(params, dict(lr=lr))
#
#     def step(self, closure=None):
#         loss = None
#
#         for group in self.param_groups:
#
#             for p in group['params']:
#
#                 if p.grad is None:
#                     continue
#
#                 gradient = p.grad.data
#                 size = gradient.size()
#                 noise = Normal(torch.zeros(size), torch.ones(size) * np.sqrt(group['lr']))
#                 gradient_w_noise = gradient + noise.sample()
#                 p.data.add_(gradient_w_noise, alpha=-group['lr'])
#
#         return loss


# new optimizer to suit the noise variance of the MNIST experiment
class NewSGLD(Optimizer):
    def __init__(self, params, lr, std_coef, device):
        super(NewSGLD, self).__init__(params, {'lr': lr})
        self.std_coef = std_coef
        self.device = device

    def step(self, closure=None):  # we could totally factor out lr and create the distribution only once, but this looks nicer
        loss = None

        for group in self.param_groups:
            for p in group['params']:

                if p.grad is None:
                    continue

                gradient = p.grad.data  # is this correct
                p.data.add_(gradient, alpha=-group['lr'])

                if self.std_coef != 0:
                    size = gradient.size()
                    noise = Normal(torch.zeros(size, device=self.device),
                                   torch.ones(size, device=self.device) * group['lr'] * self.std_coef / math.sqrt(2))
                    p.data.add_(noise.sample())

        return loss  # what even is this
