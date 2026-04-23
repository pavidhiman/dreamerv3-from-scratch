### Autograd 
- A NN is a chain of math operations: add a bias, apply a function, repeat 
- to train it, we just have to know if we nudge a weight then how would the final loss change (ie, a gradient)
- Computing gradients by hand for thousands of params is impossible so we have a system which automatically tracks operations and computes gradients - its called Autograd 

**How autograd works:**
1. Forward pass: compute the result, but also remember what operation produced each value and what the inputs were
2. Backward pass: start from the loss, walk backward through the chain, applying the chain rule at each step

### File: nn/tensor.py
- All of our data lives in NumPy arrays and our Tensor class adds gradient tracking - thus, every number in our NN will be a tensor 
- replacement for using pytorch 

### File: nn/linear.py
- A Linear layer does one thing: output = input @ weights + bias
- Also needs its own weights and biases as trainable parameters and provide a way to collect them for the optimizer 