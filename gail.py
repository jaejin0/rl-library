# GAIL : Generative Adversarial Imitation Learning

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
from collections import deque

class Generator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Generator, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Mean and standard deviation for continious actions
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.network(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        std = torch.exp(log_std)
        return mean, std

class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Discrminator, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim + action-dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, state, action):
        x = torch.cat([state, action]), dim=-1)
        return self.network(x)

class ExpertBuffer:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action):
        self.buffer.append((state, action))

    def sample(self, batch_size):
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        states, actions = zip(*[self.buffer[idx] for idx in indices])
        return torch.FloatTensor(states), torch.FloatTensor(actions)


class GAIL:
    def __init__(self, env, state_dim, action_dim, expert_buffer):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.expert_buffer = expert_buffer

        self.policy = Generator(state_dim, action_dim)
        self.discriminator = Discriminator(state_dim, action_dim)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=3e-4)

        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.disc_steps = 5

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        mean, std = self.policy(state)
        dist = Normal(mean, std)
        action = dist.sample()
        return action.detach().numpy()[0]

    def compute_discriminator_reward(self, state, action):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = torch.FloatTensor(action).unsqueeze(0)
        reward = -torch.log(self.discriminator(state, action) + 1e-8)
        return reward.item()

    def update_discriminator(self, states, actions, expert_states, expert_actions):
        policy_d = self.discrminator(states, actions)
        expert_d = self.discrminator(expert_states, expert_actions)

        disc_loss = -(torch.log(expert_d + 1e-8).mean() + torch.log(1 - policy_d + 1e-8).mean())

        self.discrminator_optimizer.zero_grad()
        disc_loss.backward()
        self.discriminator_optimizer.step()

        return disc_loss.item()

    def update_generator(self, states, actions, advantages):
        mean, std = self.policy(states)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(-1)

        policy_loss = -(log_probs * advantages).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return policy_loss.item()

    def train(self, num_episodes=1000, steps_per_episodes=1000):
        for episode in range(num_episodes):
            states, actions, rewards = [], [], []
            state = self.env.reset()

            for step in range(steps_per_episode):
                action = self.select_action(state)
                next_state, _, done, _ = self.env.step(action)

                reward = self.compute_discriminator_reward(state, action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                if done:
                    break

                state = next_state

            states = torch.FloatTensor(states)
            actions = torch.FloatTensor(actions)
            rewards = torch.FloatTensor(rewards)

            advantages = rewards

            expert_states, expert_actions = self.expert_buffer.sample(len(states))
            for _ in range(self.disc_steps):
                disc_loss = self.update_discriminator(states, actions, expert_states, expert_actions)

            policy_loss = self.update_policy(states, actions, advantages)

            if episode % 10 == 0:
                print(f"Episode {episode}")
                print(f"Discriminator Loss: {disc_loss:.4f}")
                print(f"Policy Loss: {policy_loss:.4f}")
                print(f"Average Reward: {rewards.mean().item():.4f}")
                print("---------------------------")


def main():
    env = env.make('CarRacing-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    expert_buffer = ExpertBuffer()

    gail = GAIL(env, state_dim, action_dim, expert_buffer)
    gail.train

if __name__ == "__main__":
    main()
