# basic MLP - just wrapping it in a class for RSSM
from nn.mlp import MLP

class Encoder:
    def __init__(self, obs_size, hidden_size=400, latent_size=200): # obs_size — how many sensor readings the robot has (will change)
        self.net = MLP([obs_size, hidden_size, hidden_size, latent_size]) # 2 hidden layers 
    def __call__(self, obs):
        return self.net(obs)
    def parameters(self): # run all weights for optimizer to adjust 
        return self.net.parameters()