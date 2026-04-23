import numpy as np
from nn.tensor import Tensor

class Linear:
    def __init__(self, in_features, out_features, bias=True): # in_features = input size, out_features = output size
        scale = (2.0 / in_features) ** 0.5 # weight matrix (ensure numbers are in reasonable range for training)
        self.weight = Tensor(
            np.random.randn(in_features, out_features).astype(np.float32) * scale,
            requires_grad=True,
        )
        self.bias = None
        if bias:
            self.bias = Tensor(
                np.zeros(out_features, dtype=np.float32),
                requires_grad=True,
            )

    def __call__(self, x): # output = input × weights + bias
        out = x.matmul(self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out

    def parameters(self): # Returns the list of things that need adjusting during training (the weights and biases)
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params