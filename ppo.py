import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
import gymnasium as gym
import os

'''
Proximal Policy Optimization (PPO) is an on-policy, policy gradient method.
'''

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = 'cpu'
print("device: ", device)

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.dones = []

    def clear(self):
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.state_values.clear()
        self.dones.clear()

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, initial_action_std):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
            nn.Tanh() # limits the action output between [-1.0, 1.0] 
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.action_dim = action_dim
        self.set_action_std(initial_action_std)

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def act(self, observation):
        action_mean = self.actor(observation)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean.cpu(), cov_mat.cpu()) # gaussian distribution

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(observation)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = MultivariateNormal(action_mean.cpu(), cov_mat.cpu())

        action_logprobs = dist.log_prob(action.cpu()).to(device)
        dist_entropy = dist.entropy().to(device)
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, observation_dim, action_dim, lr_actor, lr_critic, 
                 discount_factor, eps_clip, K_epochs, initial_action_std):
        self.observation_dim = observation_dim
        self.action_dim = action_dim 

        self.policy = ActorCritic(observation_dim, action_dim, initial_action_std).to(device)
        self.policy_old = ActorCritic(observation_dim, action_dim, initial_action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.AdamW([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        self.MSE_loss = nn.MSELoss()

        self.discount_factor = discount_factor
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.action_std = initial_action_std

        self.buffer = RolloutBuffer()

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        self.action_std = max(min_action_std, self.action_std)

        self.set_action_std(self.action_std)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        
        return action.detach().cpu().numpy().flatten()

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.buffer.rewards), reversed(self.buffer.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.discount_factor * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        advantages = rewards.detach() - old_state_values.detach()

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            surrogate_loss1 = ratios * advantages
            surrogate_loss2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            loss = -torch.min(surrogate_loss1, surrogate_loss2) + 0.5 * self.MSE_loss(state_values, rewards) - 0.01 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

if __name__ == "__main__": 

    # configuration
    env_name = "BipedalWalker-v3"

    max_total_steps = int(3e6)
    max_steps = 1000
    
    lr_actor = 0.0001
    lr_critic = 0.001
    
    discount_factor = 0.99
    eps_clip = 0.2 
    K_epochs = 80
    
    initial_action_std = 0.6
    action_std_decay = 0.05
    action_std_min = 0.1
    
    update_freq = max_steps * 4
    action_std_decay_freq = int(2.5e5)
    save_model_freq = int(1e5)
    
    directory = "pretrained_models"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # initialization
    env = gym.make(env_name, render_mode="human")
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    agent = PPO(observation_dim, action_dim, lr_actor, lr_critic, 
                discount_factor, eps_clip, K_epochs, initial_action_std) 

    reward_logs = []
    t = 0
    while t <= max_total_steps:
        observation, info = env.reset()
        current_ep_reward = 0
        for step in range(max_steps):
            action = agent.select_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            
            agent.buffer.rewards.append(reward)
            agent.buffer.dones.append(terminated)

            t += 1
            current_ep_reward += reward
           
            if t % update_freq == 0: 
                agent.update()

            if t % action_std_decay_freq == 0:
                agent.decay_action_std(action_std_decay, action_std_min)

            if t % save_model_freq == 0:
                checkpoint_path = directory + f"/PPO_{env_name}_timestep_{t}.pth"
                agent.save(checkpoint_path)

            if terminated or truncated:
                break 

        reward_logs.append(current_ep_reward)
        print(f"t: {t}, {np.mean(current_ep_reward)}") 
        print("action std: ", agent.action_std)

    print("Learning is done!")    
    while True:
        observation, info = env.reset() 
        for step in range(max_steps):
            action = agent.select_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

    env.close()
