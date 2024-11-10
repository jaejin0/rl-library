# lbf environment library
import gymnasium as gym
import lbforaging
import time

import numpy as np

# Joint-Action Learning with Game Theory 

# JAL-GT have variations, such as Minimax-Q, Nash-Q, and CE-Q. This program used NashQ to converge to Nash equilibrium 
# For each state of the game, create a normal-form game with Q values.
# solution concepts:
# Minimax : solution concept for a two-agent zero-sum game. 
# Nash Equilibrium (NE) : solution concept for a any number of agents general-sum game.
# Correlated Equilibrium (CE) : NE with a countable probability space, as expected value can be greater than NE.

# JAL-GT pseudocode
# // Algorithm controls agent i
# Initialize: Q_j(s,a) = 0 for all j in I and s in S, a in A
# Repeat for every episode:
# for t = 0, 1, 2, ... do
#     Observe current state s^t
#     With probability epsilon: choose random action a^t_i
#     Otherwise: solve Gamma_{s^t} to get policies (pi_1, ..., pi_n), then sample action a^t_i ~ pi_i
#     Observe joint action a^t = (a^t_1, ..., a^t_n), rewards r^t_1, ..., r^t_n, next state s^{t+1}
#     for all j in I do
#         Q_j(s^t, a^t) <- Q_j(s^t, a^t) + alpha[r^t_j + gamma * Value_j(Gamma_{s^{t+1}}) - Q_j(s^t, a^t)]

class MinimaxQ:
    # 5 inputs: 
    def __init__(self, num_agents, state_space, action_space, discount_factor, learning_rate, exploration_parameter):
        self.num_agents = num_agents
        self.state_space = state_space
        self.action_space = action_space
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.exploration_parameter = exploration_parameter

        # defining normal form game for each state
        self.action_value_function = [np.zeros([len(state_space), len(action_space)]) for i in range(self.num_agents)]
         

if __name__ == '__main__':

    # configuration:
    num_agents = 2
    discount_factor = 0.99
    learning_rate = 0.1
    exploration_parameter = 0.2 

    env = gym.make("Foraging-8x8-2p-1f-v3")
    obs = env.reset()

    max_steps = 1000
    for _ in range(max_steps):
        time.sleep(0.5)
        actions = env.action_space.sample()
        print(env.step(actions))
        env.render()
          

