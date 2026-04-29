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

### File: nn/gru_cell.py
- Everything with the MLP so far has no memory 
- GRU (Gated Recurrent Unit) is a layer that has a hidden state — a vector of numbers that it carries forward from one timestep to the next
- This is the deterministic memory of the RSSM (h_t specifically)
- GRU decides to remember by:
    - Has 2 gates:
        1. Reset gate (r) - if r = 0: ignore the past states completely, if r = 1 then use all of it
        2. Update gate (z) - if z = 1, keep the old info unchanged (nothing new). If z = 0, throw out the old info and use entirely new content

Entire concept of the GRU:
1. r = sigmoid(input × W_r + old_h × U_r)        ← how much past to consider
2. z = sigmoid(input × W_z + old_h × U_z)        ← how much to keep vs replace
3. candidate = tanh(input × W_n + (r * old_h) × U_n)  ← proposed new content
4. new_h = z * old_h + (1 - z) * candidate       ← blend old and new

### File: model/encoder.py
- World models need encoders since if a robot has ~300 sensor readings - theres overlap between some of these (ex, left knee is at 30 degrees and the left thigh is at 45 degrees, you could probably guess the left ankle position pretty well)
- Encoder compresses these 300 numbers into maybe 200 (keeping important stuff) for 2 reasons:
    1. GRU (memory) must carry these numbers forward so smaller = faster = easier to learn
    2. Forces the network to identify what actually matters
Overall flow: raw sensors (300) → [ENCODER] → compressed (200) → [RSSM] → internal state → [DECODER] → reconstructed sensors (300)

### File; model/rssm.py (KEY part of how this runs)
- To predict the future state based on the robots current state and current action, theres 2 challenges:
    1. Memory - GRU handles this by knowing the current state and all previous states. 
        => hidden state 'h'
    2. Uncertainty - even with perfect memory, you can't perfectly predcit the next step since a joint might have friction, foot might slip, etc. 
        - By predicting only 1 future, it'll learn an average which is useless
        - Solution: predict a distribution (ex, 60% chance of this, 30% of this, etc) and then sample from this 
        => stochastic latent 'z'

**How 'z' works**
- z is represented as 32 categorical variables with 32 classes (like rolling 32 dice each with 32 faces)
    - At each timestep, model rolls all 32 dice and the combination of results represents one specific possibility of whats happening
- z is a tensor of shape (batch, 32, 32) — 32 dice, each with 32 probabilities. We flatten it to (batch, 1024) when feeding it into other layers


**2 paths for producing z**
1. Posterior (used during training) - by checking the real sensor readings, you can extrapolate what z should be 
    Input: h (memory) + encoded observation (real data)
    Output: z (accurate)

2. Prior (used during imagination) - using memory, guesses what z should be 
    Input: h (memory) only
    Output: z (best guess)

Point of the training is to make priors guess match the posterior's accurate answer 
- KL divergence loss will measrure how different they are and pushes them together 
- Eventually, once the prior is good enough, model can imagine the future without real data 

Flow of the RSSM:
1. Take previous h, previous z, previous action
2. Concatenate them: [z, action] → feed into GRU along with h
3. GRU outputs new h (updated memory)
4. Prior: MLP takes h → predicts z probabilities (the guess)
5. Posterior: MLP takes [h, encoded_obs] → predicts z probabilities (the answer)
6. During training: use posterior's z (the accurate one)
   During imagination: use prior's z (the guess)
7. Sample z from the probabilities (roll the dice)
8. Full state = [h, z] — this is what decoders/reward predictors look at

### File: model/decoder.py
- Takes internal state and tries to reconstruct the original sensor readings - if the decoder accurately reconstructs reality from the internal state, it proves there's useful information there
- Prevents RSSM to learn useless internal states 

### File: model/reward_model.py
- Takes the internal state and predicts "how much reward will the robot get in this state?"
     - During imagination, this is how the robot knows if an imagined future is good or bad 

### File: model/continue.py
- Takes the internal state and predicts "is the episode still going, or did the robot fall over?"
    - Output is a probability between 0 (episode over) and 1 (still going) 

### File: model/world_model.py
- Currently have 5 separate pieces: encoder, decoder, RSSM, reward model, continue model 
- To train the world model we must pass data through them in the right order, compute all losses and collect all params for the optimizer - this file allows us to call world_model.train_step(data) and computes the whole pipeline

**During 1 training step:**
For each timestep in the sequence:
  1. Encode the observation          (encoder)
  2. Run the RSSM forward            (RSSM: GRU + posterior + sample z)
  3. Decode the state back            (decoder → reconstruction loss)
  4. Predict the reward               (reward model → reward loss)
  5. Predict continue                 (continue model → continue loss)

Then also:
  6. KL loss between prior and posterior (forces the prior to learn to guess well)
  The total loss is: reconstruction + reward + continue + KL

### Files: agent/actor.py and agent/critic.py
- This is for the RL agent that learns to walk inside the world models imagination 
- Actor - looks at the current state and picks an action.
    - Outputs a mean and st. dev for each action and then samples from this distribution 
- Critic - looks at the current state and estimates "how much total future reward will I get from here?" It's like a score predictor

**How they train together in imagination:** 
1. Start from a real state the world model has seen
2. Actor picks an action
3. World model imagines what happens next (using prior, no real data)
4. Critic scores each imagined state
5. Actor adjusts to pick actions that lead to higher-scored states
6. Critic adjusts to score more accurately
The actor is trying to maximize the critic's score