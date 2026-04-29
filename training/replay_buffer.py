import numpy as np
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity # max num of timesteps to store
        self.observations = []
        self.actions = []
        self.rewards = []
        self.continues = []
    def add(self, obs, action, reward, done):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.continues.append(0.0 if done else 1.0)
        if len(self.observations) > self.capacity:
            self.observations.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.continues.pop(0)
    def sample(self, batch_size, seq_len):
        max_start = len(self.observations) - seq_len
        if max_start < 1:
            return None
        indices = np.random.randint(0, max_start, size=batch_size)
        obs_seqs = []
        act_seqs = []
        rew_seqs = []
        cont_seqs = []
        for t in range(seq_len):
            obs_seqs.append(np.stack([self.observations[i + t] for i in indices]).astype(np.float32))
            act_seqs.append(np.stack([self.actions[i + t] for i in indices]).astype(np.float32))
            rew_seqs.append(np.stack([[self.rewards[i + t]] for i in indices]).astype(np.float32))
            cont_seqs.append(np.stack([[self.continues[i + t]] for i in indices]).astype(np.float32))
        return obs_seqs, act_seqs, rew_seqs, cont_seqs
    def __len__(self):
        return len(self.observations)