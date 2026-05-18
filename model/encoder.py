from nn.mlp import MLP

class Encoder:
    def __init__(self, obs_size, hidden_size, latent_size):
        self.net = MLP([obs_size, hidden_size, hidden_size, latent_size])
    def __call__(self, obs): return self.net(obs)
    def parameters(self): return self.net.parameters()
