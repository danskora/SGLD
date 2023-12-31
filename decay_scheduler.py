__all__ = ['DecayScheduler']


class DecayScheduler:

    def __init__(self, optimizer, lr, rate, step_size, floor=None):
        self.optimizer = optimizer
        self.lr = lr
        self.rate = rate
        self.step_size = step_size
        self.floor = floor
        self.step_count = 1

        for group in self.optimizer.param_groups:
            group['lr'] = lr

    def step(self):

        self.step_count += 1

        if self.step_count % self.step_size == 0:
            self.lr *= self.rate
            if self.floor and self.lr < self.floor:
                self.lr = self.floor
            for group in self.optimizer.param_groups:
                group['lr'] = self.lr

