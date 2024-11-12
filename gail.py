# GAIL : Generative Adversarial Imitation Learning

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

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



class Discriminator():
    '''
    input : camera, speed, target, command, action
        camera : 9x144x256 (3 cameras with 3 colors)
        speed : 1x1
        target : 2x1
        command : 6x1
        action : 2x1

    output : V(s), steer, throttle, D(s, a)
        V(s) : 1x1 (current state value function)
        steer : 1x1
        throttle : 1x1
        D(s, a) : 1x1
    '''


