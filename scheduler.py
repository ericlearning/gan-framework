import math
from utils.network_utils import get_total_epoch


class LinearDecay():
    def __init__(self, opt, optimizer, iter_num):
        self.optimizer = optimizer
        self.inits = [p['lr'] for p in self.optimizer.param_groups]
        self.tot, self.st = iter_num * epoch, iter_num * decay_start_epoch
        self.cnt = 0

    def step(self):
        for p, lr in zip(self.optimizer.param_groups, self.inits):
            if(self.cnt < self.st):
                p['lr'] = lr
            else:
            	n = (self.cnt - self.st) / (self.tot - 1 - self.st)
                p['lr'] = lr * (1.0 - n)
        self.cnt += 1
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()
