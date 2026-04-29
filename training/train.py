import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym
from nn.tensor import Tensor, cat
from nn.optimizer import AdamW
from model.world_model import WorldModel
from agent.actor import Actor
from agent.critic import Critic
from training.replay_buffer import ReplayBuffer

def collect_data(env, buffer, num_steps=1000):
    obs, _ = env.reset()
    for _ in range(num_steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add(obs, action, reward, done)
        obs = next_obs
        if done:
            obs, _ = env.reset()

def train_world_model(world_model, optimizer, buffer, batch_size=16, seq_len=16):
    batch = buffer.sample(batch_size, seq_len)
    if batch is None:
        return None
    obs_seqs, act_seqs, rew_seqs, cont_seqs = batch # sampling batch from replay buffer 
    observations = [Tensor(o) for o in obs_seqs]
    actions = [Tensor(a) for a in act_seqs]
    rewards = [Tensor(r) for r in rew_seqs]
    continues = [Tensor(c) for c in cont_seqs]
    optimizer.zero_grad() # numpy array to tensor 
    loss = world_model.forward(observations, actions, rewards, continues)
    loss.backward()
    optimizer.step()
    return float(loss.data)


def train_actor_critic(world_model, actor, critic, actor_opt, critic_opt,
                       buffer, horizon=15, batch_size=16, seq_len=16):
    batch = buffer.sample(batch_size, 1) # sample 1 single timestep to use as startng state
    if batch is None:
        return None, None
    obs_start = Tensor(batch[0][0])
    act_start = Tensor(batch[1][0])
    encoded = world_model.encoder(obs_start)
    h, z, _, _ = world_model.rssm.forward(None, None, act_start, encoded_obs=encoded)
    h = h.detach() # don't want imagination gradients flowing back into world model 
    z = z.detach()
    imagined_h = [h]
    imagined_z = [z]
    rewards = []
    continues = []
    for _ in range(horizon):
        action = actor(h, z)
        h, z, _, _ = world_model.rssm.forward(h, z, action, encoded_obs=None)
        rewards.append(world_model.reward_model(h, z))
        continues.append(world_model.continue_model(h, z))
        imagined_h.append(h)
        imagined_z.append(z)
    values = [critic(h, z) for h, z in zip(imagined_h, imagined_z)]
    lambda_val = 0.95
    returns = [values[-1]]
    for t in reversed(range(horizon)):
        r = rewards[t]
        c = continues[t]
        v_next = returns[0]
        ret = r + c * 0.99 * ((1 - lambda_val) * values[t + 1] + lambda_val * v_next)
        returns.insert(0, ret)
    returns = returns[:-1]
    actor_loss = Tensor(0.0)
    critic_loss = Tensor(0.0)
    for t in range(horizon):
        actor_loss = actor_loss + (-returns[t]).mean()
        critic_loss = critic_loss + ((values[t] - returns[t].detach()) ** 2).mean()
    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()
    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()
    return float(actor_loss.data), float(critic_loss.data)


def evaluate(env, world_model, actor, num_episodes=5):
    total_rewards = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        h, z = None, None
        episode_reward = 0
        done = False
        steps = 0
        while not done and steps < 200:
            obs_tensor = Tensor(obs.reshape(1, -1).astype(np.float32))
            encoded = world_model.encoder(obs_tensor)
            prev_action = Tensor(np.zeros((1, env.action_space.shape[0]), dtype=np.float32))
            h, z, _, _ = world_model.rssm.forward(h, z, prev_action, encoded_obs=encoded)
            h_det = h.detach()
            z_det = z.detach()
            action = actor(h_det, z_det)
            act_np = np.clip(action.data[0], -1, 1) * 2.0
            obs, reward, terminated, truncated, _ = env.step(act_np)
            episode_reward += reward
            done = terminated or truncated
            steps += 1
        total_rewards.append(episode_reward)
    avg = sum(total_rewards) / len(total_rewards)
    return avg


if __name__ == '__main__':
    env = gym.make('Pendulum-v1')
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    hidden_size = 64
    stoch_size = 8
    stoch_classes = 8
    state_size = hidden_size + stoch_size * stoch_classes
    world_model = WorldModel(obs_size, action_size, hidden_size, stoch_size, stoch_classes)
    actor = Actor(state_size, action_size, hidden_size=64)
    critic = Critic(state_size, hidden_size=64)
    wm_opt = AdamW(world_model.parameters(), lr=3e-4)
    actor_opt = AdamW(actor.parameters(), lr=3e-4)
    critic_opt = AdamW(critic.parameters(), lr=3e-4)
    buffer = ReplayBuffer()

    print('Collecting initial data...')
    collect_data(env, buffer, num_steps=1000)
    print(f'Buffer size: {len(buffer)}')

    random_reward = evaluate(env, world_model, actor, num_episodes=3)
    print(f'Before training: avg reward = {random_reward:.1f}')

    for step in range(500):
        wm_loss = train_world_model(world_model, wm_opt, buffer, batch_size=8, seq_len=8)
        ac_result = train_actor_critic(
            world_model, actor, critic, actor_opt, critic_opt,
            buffer, horizon=8, batch_size=8,
        )
        if step % 100 == 0:
            avg_reward = evaluate(env, world_model, actor, num_episodes=3)
            print(f'step {step}: wm_loss={wm_loss:.4f}, eval_reward={avg_reward:.1f}')

    final_reward = evaluate(env, world_model, actor, num_episodes=5)
    print(f'After training: avg reward = {final_reward:.1f}')
    print('Training complete.')