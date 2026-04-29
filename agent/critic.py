from nn.mlp import MLP
from nn.tensor import cat

class Critic:
    def __init__(self, state_size, hidden_size=400):
        self.net = MLP([state_size, hidden_size, hidden_size, 1])

    def __call__(self, h, z):
        state = cat([h, z], axis=1)
        return self.net(state)

    def parameters(self):
        return self.net.parameters()