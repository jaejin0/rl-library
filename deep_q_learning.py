import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
import random

'''
difference between Deep Q Learning and Deep Q Network

Deep Q Learning uses a single neural network for its Q function and updates its policy network for every iteration

Deep Q Network uses two neural network for its Q function and target network, to solve the "moving target problem"
by providing stationary of the target.
Q function or Q network is trained more often than target network.
'''

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return states, actions, rewards, next_states, dones

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
        action_prob = self.network(observation)
        action_prob = F.softmax(action_prob, dim=0)
        return action_prob

class DeepQLearning:
    def __init__(self, observation_dim, action_dim, exploration_parameter, discount_factor, learning_rate, buffer_size, batch_size):
        self.observation_dim = observation_dim
        self.action_dim = action_dim 
       
        self.exploration_parameter = exploration_parameter
        self.discount_factor = discount_factor

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = 'cpu'
        print("device: ", self.device)

        self.value_network = QNetwork(observation_dim, action_dim).to(self.device)
        self.loss_function = nn.SmoothL1Loss()
        self.optimizer = torch.optim.AdamW(self.value_network.parameters(), lr=learning_rate) 
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

    def policy(self, observation):
        explore = np.random.binomial(1, self.exploration_parameter)
        if explore:
            action = random.randrange(self.action_dim)
            action = torch.tensor(action).to(self.device)
        else: # exploit
            with torch.no_grad():
                action_prob = self.value_network.forward(observation) 

            # instead of argmax, I used a random-action approach
            # action = np.random.choice(range(self.action_dim), p=action_prob)
            # action = action_prob.argmax() 
            action = action_prob.multinomial(num_samples=1, replacement=True).squeeze() 
        return action

    def learn(self):
        observations, actions, rewards, next_observations, dones = self.replay_buffer.sample(self.batch_size)
        observations = torch.stack(observations)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_observations = torch.stack(next_observations)
        dones = torch.tensor(dones).float().to(self.device)

        with torch.no_grad():
            next_max_q = self.value_network(next_observation)
            next_max_q = torch.max(next_max_q)
            # target_value = rewards + (1 - dones) * self.discount_factor * next_max_q.item()
            target_value = rewards + self.discount_factor * next_max_q.item()
        current_value = self.value_network(observations).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        
        loss = self.loss_function(current_value, target_value)
        
        self.optimizer.zero_grad() # clear gradients
        loss.backward() # compute gradients (backpropagation)
        self.optimizer.step() # update network parameters

if __name__ == "__main__": 
    # configuration
    num_episodes = 1000
    max_steps = 1000
    exploration_parameter = 0.9
    exploration_end = 0.05
    exploration_decay = 0.997
    discount_factor = 0.99
    learning_rate = 0.0001
    buffer_size = 10000
    batch_size = 128
    # initialization
    env = gym.make("CartPole-v1", render_mode="human")
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DeepQLearning(observation_dim, action_dim, exploration_parameter, discount_factor, learning_rate, buffer_size, batch_size) 


    reward_logs = []
    for ep in range(num_episodes):
        observation, info = env.reset()
        observation = torch.tensor(observation).to(agent.device)
        total_reward = 0

        for step in range(max_steps):
            # action = env.action_space.sample()
            action = agent.policy(observation)
            next_observation, reward, terminated, truncated, info = env.step(action.item())
            
            next_observation = torch.tensor(next_observation).to(agent.device)
            reward = torch.tensor(reward).to(agent.device)

            agent.replay_buffer.push(observation, action, reward, next_observation, terminated) # or truncated) truncated is not included as it does not count make reward 0
            
            if len(agent.replay_buffer) >= batch_size:
                agent.learn()
            
            # transition
            observation = next_observation
            if agent.exploration_parameter >= exploration_end:
                agent.exploration_parameter *= exploration_decay
            total_reward += reward.item()
            
            if terminated or truncated:
                break 
        
        reward_logs.append(total_reward)

        if ep % 10 == 0:
            print(ep, ": average reward of last 10 episodes: ", np.mean(reward_logs[-10:]))



    print("Learning is done!")    
    while True:
        observation, info = env.reset() 
        for step in range(max_steps):
            action = agent.policy(torch.tensor(observation).to(agent.device))
            observation, reward, terminated, truncated, info = env.step(action.item())
            if terminated or truncated:
                break

    env.close()
