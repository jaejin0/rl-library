import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return (np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones))

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, observation):
        return self.network(observation)

class DQN:
    def __init__(self, observation_dim, action_dim, batch_size, buffer_size, exploration_parameter):
        self.observation_dim = observation_dim
        self.action_dim = action_dim 
        self.batch_size = batch_size
        
        self.exploration_parameter = exploration_parameter

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = 'cpu'
        print("device: ", self.device)

        self.value_network = QNetwork(observation_dim, action_dim).to(self.device)
        self.target_network = QNetwork(observation_dim, action_dim).to(self.device)
        self.replay_buffer = ReplayBuffer(buffer_size)
   
    def policy(self, observation):
        explore = np.random.binomial(1, self.exploration_parameter)
        if explore:
            action = random.randrange(self.action_dim)
        else: # exploit
            observation = torch.from_numpy(observation)
            observation = observation.to(self.device)
            action_prob = self.value_network(observation)
            action = np.random.choice(range(self.action_dim), p=action_prob)
        print(action)

if __name__ == "__main__": 
    # configuration
    num_episodes = 1000
    max_steps = 1000
    batch_size = 64
    buffer_size = 10000

    exploration_parameter = 0.2

    # initialization
    env = gym.make("CartPole-v1", render_mode="human")
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN(observation_dim, action_dim, batch_size, buffer_size, exploration_parameter) 

    for ep in range(num_episodes):
        observation, info = env.reset()
        total_reward = 0

        for step in range(max_steps):
            # action = env.action_space.sample()
            action = agent.policy(observation)

            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
               break
        print(ep, total_reward) 
    env.close()
