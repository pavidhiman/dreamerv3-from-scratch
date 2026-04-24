import numpy as np
from nn.linear import Linear
from nn.tensor import Tensor, cat

class GRUCell:
    def __init__(self, input_size, hidden_size): # hidden_size = size of history   
        self.hidden_size = hidden_size
        self.reset_gate = Linear(input_size + hidden_size, hidden_size)
        self.update_gate = Linear(input_size + hidden_size, hidden_size)
        self.candidate = Linear(input_size + hidden_size, hidden_size)

    def __call__(self, x, h): # forward pass: taking new input x and old hidden state h, and outputting new hidden state h_new
        if h is None:
            h = Tensor(np.zeros((x.shape[0], self.hidden_size), dtype=np.float32))
        combined = cat([x, h], axis=1)
        r = self.reset_gate(combined).sigmoid()
        z = self.update_gate(combined).sigmoid()
        h_reset = r * h
        combined_r = cat([x, h_reset], axis=1)
        candidate = self.candidate(combined_r).tanh()
        h_new = z * h + (1 - z) * candidate
        return h_new

    def parameters(self):
        params = []
        for layer in [self.reset_gate, self.update_gate, self.candidate]:
            params.extend(layer.parameters())
        return params