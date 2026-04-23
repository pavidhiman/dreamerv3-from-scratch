''' 
MLP = stacking several Linear layers with activations between them 
example: input → Linear(300, 400) → silu → Linear(400, 400) → silu → Linear(400, 200) → output
that takes 300 inputs, passes through two hidden layers of size 400 with SiLU activation, and produces 200 outputs

every component in DreamerV3 is an MLP (encoder, decoder, actor, critic, etc)
''' 

from nn.linear import Linear
from nn.tensor import Tensor
class MLP:
    def __init__(self, sizes, activation='silu'):
        self.activation = activation
        self.layers = []
        # Create the Linear layers. If sizes = [300, 400, 400, 200], this creates three Linear layers:
        # Linear(300, 400), Linear(400, 400), Linear(400, 200). Each consecutive pair of 
        # sizes defines one layer
        for i in range(len(sizes) - 1):
            self.layers.append(Linear(sizes[i], sizes[i + 1]))

    # forward pass - passes data through each layer except last one (no activation after last layer)
    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation == 'silu':
                    x = x.silu()
                elif self.activation == 'relu':
                    x = x.relu()
        return x

    # returns all the trainable parameters in the MLP (all the weights and biases)
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params