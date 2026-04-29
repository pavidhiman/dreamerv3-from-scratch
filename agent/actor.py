import numpy as np
from nn.mlp import MLP
from nn.tensor import Tensor, cat
class Actor:
    def __init__(self, state_size, action_size, hidden_size=400):
        self.action_size = action_size # num of joint torques 
        self.net = MLP([state_size, hidden_size, hidden_size, action_size * 2])
    def __call__(self, h, z):
        state = cat([h, z], axis=1)
        action = self.net(state).tanh()
        return action
    def parameters(self):
        return self.net.parameters()