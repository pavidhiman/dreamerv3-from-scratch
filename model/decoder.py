from nn.mlp import MLP
from nn.tensor import cat
class Decoder:
    def __init__(self, state_size, obs_size, hidden_size=400):
        self.net = MLP([state_size, hidden_size, hidden_size, obs_size])
    def __call__(self, h, z):
        state = cat([h, z], axis=1) # concatenate h and z into 1 vector
        return self.net(state)
    def parameters(self):
        return self.net.parameters()