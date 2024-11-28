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
    def __init__(self, observation_dim, action_dim, exploration_parameter, discount_factor, learning_rate):
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
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.value_network.parameters(), learning_rate) 

    def policy(self, observation):
        explore = np.random.binomial(1, self.exploration_parameter)
        if explore:
            action = random.randrange(self.action_dim)
        else: # exploit
            observation = torch.from_numpy(observation)
            observation = observation.to(self.device)
            action_prob = self.value_network(observation).cpu().detach() 
            action_prob = action_prob.numpy()

            # instead of argmax, I used a random-action approach
            action = np.random.choice(range(self.action_dim), p=action_prob)
        return action

    def learn(self, observation, action, reward, next_observation, done):
 
        action_prob = self.value_network(observation).cpu().detach()
        action_prob = action_prob.numpy()
        
        if done:
            target_value = reward
        else: 
            target_prob = self.value_network(next_observation).cpu().detach()
            traget_prob = target_prob.numpy()

        loss = self.loss_function(action_prob, target_prob)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__": 
    # configuration
    num_episodes = 1000
    max_steps = 1000
    exploration_parameter = 0.1
    discount_factor = 0.5
    learning_rate = 0.001

    # initialization
    env = gym.make("CartPole-v1", render_mode="human")
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DeepQLearning(observation_dim, action_dim, exploration_parameter, discount_factor, learning_rate) 

    for ep in range(num_episodes):
        observation, info = env.reset()
        total_reward = 0

        for step in range(max_steps):
            # action = env.action_space.sample()
            action = agent.policy(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            agent.learn(observation, action, reward, next_observation, terminated or truncated)
            observation = next_observation
            total_reward += reward
            if terminated or truncated:
                # target y^t <- r^t
                break 
            else:
                # target y^t <- r^t + gamma Q(s^{t+1}, a'; theta)
                pass
            # compute loss
            # train the model

        print(ep, total_reward)





    print("Learning is done!")    
    while True:
        observation, info = env.reset()
        for step in range(max_steps):
            action = agent.policy(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or trunctated:
                break

    env.close()
