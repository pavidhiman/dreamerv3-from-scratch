import numpy as np
class AdamW:
    def __init__(self, parameters, lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0 # counter for updates 
        self.m = [np.zeros_like(p.data) for p in self.parameters] # momentum vector - avg of most recent gradients
        self.v = [np.zeros_like(p.data) for p in self.parameters] # vector - avg of recent squared gradients
