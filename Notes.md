### Autograd 
- A NN is a chain of math operations: add a bias, apply a function, repeat 
- to train it, we just have to know if we nudge a weight then how would the final loss change (ie, a gradient)
- Computing gradients by hand for thousands of params is impossible so we have a system which automatically tracks operations and computes gradients - its called Autograd 

**How autograd works:**
1. Forward pass: compute the result, but also remember what operation produced each value and what the inputs were
2. Backward pass: start from the loss, walk backward through the chain, applying the chain rule at each step

Basics:
- example: let's say you have 3 sensors and want 1 output: output = sensor1 * w1 + sensor2 * w2 + sensor3 * w3
- how does it learn?
    1. correct answer to compare it against (this is the loss, we want the loss to be 0)
    2. way to know which direction to adjust each weight = gradient 
        - The gradient of w1 tells you: "if you increase w1 a tiny bit, does the loss go up or down?
        - If the gradient is positive → increasing w1 makes the loss worse → decrease w1
        - If the gradient is negative → increasing w1 makes the loss better → increase w1

forward and backward pass:
1. Forward pass = plugging in the sensor numbers and computing the output. Data flows forward: inputs → output. This is just doing the math.
2. Backward pass = starting from the loss and figuring out the gradient for every weight. Data flows backward: loss → output → weights. This is the chain rule from calculus, automated

- you have gradients for every weight and the optimizer is the thing which actually changes these weights 
You repeat this loop thousands of times:
Forward pass (compute output)
Compute loss (how wrong were we?)
Backward pass (compute gradients)
Optimizer step (nudge weights using gradients)
- with each loop, the loss gets a little smaller -> weights get better -> network eventually gives good answers 

### File: nn/tensor.py
- All of our data lives in NumPy arrays and our Tensor class adds gradient tracking - thus, every number in our NN will be a tensor 
- replacement for using pytorch 

### File: nn/linear.py
- A Linear layer does one thing: output = input @ weights + bias
- Also needs its own weights and biases as trainable parameters and provide a way to collect them for the optimizer 