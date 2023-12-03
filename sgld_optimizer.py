import torch
from torch.distributions import Normal
from torch.optim import Optimizer


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
                                   torch.ones(size, device=self.device))
                    p.data.add_(noise.sample(), alpha=group['lr'] * self.std_coef)

        return loss  # what even is this
