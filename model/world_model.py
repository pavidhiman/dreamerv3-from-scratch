import numpy as np
from nn.tensor import Tensor, cat
from model.encoder import Encoder
from model.rssm import RSSM
from model.decoder import Decoder
from model.reward_model import RewardModel
from model.continue_model import ContinueModel

class WorldModel:
    def __init__(self, obs_size, action_size, hidden_size=200, stoch_size=32, stoch_classes=32):
        self.stoch_size = stoch_size
        self.stoch_classes = stoch_classes
        state_size = hidden_size + stoch_size * stoch_classes # what the decoder, reward, and continue models take as input
        # all 5 components 
        self.encoder = Encoder(obs_size, latent_size=hidden_size)
        self.rssm = RSSM(hidden_size, action_size, hidden_size, stoch_size, stoch_classes)
        self.decoder = Decoder(state_size, obs_size)
        self.reward_model = RewardModel(state_size)
        self.continue_model = ContinueModel(state_size)
    def forward(self, observations, actions, rewards, continues):
        seq_len = len(observations)
        h, z = None, None
        all_h, all_z = [], []
        all_prior_logits, all_post_logits = [], []
        for t in range(seq_len): # all outputs per timestep to compute losses 
            # per timestep: encode the observation, run the RSSM (which updates memory, produces z, gives prior and posterior logits)
            encoded = self.encoder(observations[t])
            h, z, prior_logits, post_logits = self.rssm.forward(h, z, actions[t], encoded_obs=encoded)
            all_h.append(h)
            all_z.append(z)
            all_prior_logits.append(prior_logits)
            all_post_logits.append(post_logits)
        H = cat(all_h, axis=0)
        Z = cat(all_z, axis=0)
        recon = self.decoder(H, Z)
        obs_target = cat(observations, axis=0)
        recon_loss = ((recon - obs_target.symlog()) ** 2).mean() # reconstruction loss after decoder reconstructs the sensor readings
        pred_reward = self.reward_model(H, Z)
        reward_target = cat(rewards, axis=0)
        reward_loss = ((pred_reward - reward_target.symlog()) ** 2).mean() # reward loss - predict reward and compare to real
        pred_continue = self.continue_model(H, Z)
        continue_target = cat(continues, axis=0)
        continue_loss = ((pred_continue - continue_target) ** 2).mean() # continue loss - predict whether the episode is still going, compare to the real flag
        prior = cat(all_prior_logits, axis=0)
        post = cat(all_post_logits, axis=0)
        prior_probs = prior.softmax(axis=-1)
        post_probs = post.softmax(axis=-1)
        kl = (post_probs * (post_probs.log() - prior_probs.log())).sum(axis=-1).mean()
        kl_loss = kl.clamp(min_val=1.0)
        loss = recon_loss + reward_loss + continue_loss + kl_loss
        return loss
    def parameters(self):
        params = self.encoder.parameters()
        params.extend(self.rssm.parameters())
        params.extend(self.decoder.parameters())
        params.extend(self.reward_model.parameters())
        params.extend(self.continue_model.parameters())
        return params