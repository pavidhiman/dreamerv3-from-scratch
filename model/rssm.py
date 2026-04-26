import numpy as np
from nn.tensor import Tensor, cat, one_hot
from nn.linear import Linear
from nn.mlp import MLP
from nn.gru_cell import GRUCell

class RSSM:
    # stoch_size = 32 categorical variables, stoch_classes = 32 classes per var
    def __init__(self, obs_size, action_size, hidden_size=200, stoch_size=32, stoch_classes=32): 
        self.hidden_size = hidden_size
        self.stoch_size = stoch_size
        self.stoch_classes = stoch_classes
        stoch_flat = stoch_size * stoch_classes # 32 x 32 = 1024

        # before going into GRU, combine the previous z and previous action into 1 vector
        # linear takes [z_flat, action] and compresses into hidden_size (ie, what GRU expects as input)
        self.pre_gru = Linear(stoch_flat + action_size, hidden_size)
        self.gru = GRUCell(hidden_size, hidden_size)
 
        self.prior = MLP([hidden_size, hidden_size, stoch_flat]) # prior - takes GRUs h and outputs 1024 nums
        self.posterior = MLP([hidden_size + obs_size, hidden_size, stoch_flat]) # posterior - takes GRUs h and encoded obs and outputs 1024 nums
    
    def forward(self, prev_h, prev_z, action, encoded_obs=None):
        if prev_z is None:
            prev_z = Tensor(np.zeros((action.shape[0], self.stoch_size * self.stoch_classes), dtype=np.float32))
        if prev_h is None:
            prev_h = Tensor(np.zeros((action.shape[0], self.hidden_size), dtype=np.float32))
        gru_input = self.pre_gru(cat([prev_z, action], axis=1)).silu() # pass thru 1 linear layer with silu activation
        h = self.gru(gru_input, prev_h) # runs 1 GRU step -> memory updated 
        prior_logits = self.prior(h).reshape(-1, self.stoch_size, self.stoch_classes)
        if encoded_obs is not None: # if real training then use posterior 
            post_logits = self.posterior(cat([h, encoded_obs], axis=1)).reshape(-1, self.stoch_size, self.stoch_classes)
            logits = post_logits
        else: # otherwise use prior
            post_logits = None
            logits = prior_logits
        probs = logits.softmax(axis=-1) # raw -> probabilities which sum to 1 
        indices = np.array([
            [np.random.choice(self.stoch_classes, p=probs.data[b, d])
             for d in range(self.stoch_size)]
            for b in range(probs.shape[0])
        ])
        hard = one_hot(indices, self.stoch_classes)
        z = probs.straight_through(hard)
        z_flat = z.reshape(-1, self.stoch_size * self.stoch_classes)
        return h, z_flat, prior_logits, post_logits

    def parameters(self):
        params = self.pre_gru.parameters()
        params.extend(self.gru.parameters())
        params.extend(self.prior.parameters())
        params.extend(self.posterior.parameters())
        return params