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

class DQN:
    def __init__(self, input_dim, output_dim, learning_rate, gamma):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.gamma = gamma






if __name__ == "__main__":


    buffer = ReplayBuffer(1000)
    for i in range(3):
        buffer.push("s", "a", "r", "ns", "d")
    sample = buffer.sample(3) 
    print(sample)

    # configuration
    num_episodes = 1000
    max_steps = 1000

    env = gym.make("CartPole-v1", render_mode="human")
    for ep in range(num_episodes):
        observation, info = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
               break
        print(ep, total_reward) 
    env.close()
