import torch
import torch.nn as nn


class Optimizer:
    def __init__(self, params, lr, **kwargs):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    def step(self):
        raise NotImplementedError


class gd(Optimizer):
    def __init__(self, params, lr, **kwargs):
        super().__init__(params, lr, **kwargs)
    
    def step(self):
        if self.params is not None:
            for param in self.params:
                if param.grad is not None:
                    param.data -= self.lr * param.grad


class Momentum(Optimizer):
    def __init__(self, params, lr, beta ,**kwargs):
        super().__init__(params, lr, **kwargs)
        self.beta = beta
        self.vt_prev = [torch.zeros_like(p.data) for p in self.params]

    def step(self):
        with torch.no_grad():
            for i, param in enumerate(self.params):
                if param.grad is not None:
                    self.vt_prev[i] = self.beta*self.vt_prev[i] + param.grad
                    param.data -= self.lr*self.vt_prev[i]

                    