import numpy as np
from nn.linear import Linear
from nn.tensor import Tensor, cat
from cuda.backend import gru_forward as _cuda_gru_forward


class GRUCell:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.reset_gate = Linear(input_size + hidden_size, hidden_size)
        self.update_gate = Linear(input_size + hidden_size, hidden_size)
        self.candidate = Linear(input_size + hidden_size, hidden_size)

    def __call__(self, x, h):
        if h is None:
            h = Tensor(np.zeros((x.shape[0], self.hidden_size), dtype=np.float32))

        result = self._try_cuda(x, h)
        if result is not None:
            return result

        combined = cat([x, h], axis=1)
        r = self.reset_gate(combined).sigmoid()
        z = self.update_gate(combined).sigmoid()
        h_reset = r * h
        combined_r = cat([x, h_reset], axis=1)
        candidate = self.candidate(combined_r).tanh()
        h_new = z * h + (1 - z) * candidate
        return h_new

    def _try_cuda(self, x, h):
        if not x.requires_grad and not h.requires_grad:
            h_out = _cuda_gru_forward(
                x.data, h.data,
                self.reset_gate.weight.data, self.reset_gate.bias.data,
                self.update_gate.weight.data, self.update_gate.bias.data,
                self.candidate.weight.data, self.candidate.bias.data,
            )
            if h_out is not None:
                return Tensor(h_out)
        return None

    def parameters(self):
        params = []
        for layer in [self.reset_gate, self.update_gate, self.candidate]:
            params.extend(layer.parameters())
        return params