# lbf environment library
import gymnasium as gym
import lbforaging
import time

import numpy as np

# Joint-Action Learning with Game Theory 
'''
JAL-GT have variations, such as Minimax-Q, Nash-Q, and CE-Q. This program used NashQ to converge to Nash equilibrium 
For each state of the game, create a normal-form game with Q values.
solution concepts:
Minimax : solution concept for a two-agent zero-sum game. 
Nash Equilibrium (NE) : solution concept for a any number of agents general-sum game.
Correlated Equilibrium (CE) : NE with a countable probability space, as expected value can be greater than NE.

JAL-GT pseudocode
// Algorithm controls agent i
Initialize: Q_j(s,a) = 0 for all j in I and s in S, a in A
Repeat for every episode:
for t = 0, 1, 2, ... do
    Observe current state s^t
    With probability epsilon: choose random action a^t_i
    Otherwise: solve Gamma_{s^t} to get policies (pi_1, ..., pi_n), then sample action a^t_i ~ pi_i
    Observe joint action a^t = (a^t_1, ..., a^t_n), rewards r^t_1, ..., r^t_n, next state s^{t+1}
    for all j in I do
        Q_j(s^t, a^t) <- Q_j(s^t, a^t) + alpha[r^t_j + gamma * Value_j(Gamma_{s^{t+1}}) - Q_j(s^t, a^t)]
'''

# This is a MinimaxQ algorithm for a single agent, as it is independent learning, not central learning.
# you need to call this class for each agents
class MinimaxQ:
    # 6 inputs : (N, S, A, gamma, alpha, epsilon) 
    def __init__(self, num_agents, discount_factor, learning_rate, exploration_parameter):
        self.num_agents = num_agents
        
        # state space and action space doesn't have to be known for this implementation as it uses a hashmap to store values
        # self.state_space = state_space
        # self.action_space = action_space
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.exploration_parameter = exploration_parameter

        # defining normal form game for each state
        self.action_value_function = [{} for i in range(self.num_agents)]
        print(self.action_value_function) 

if __name__ == '__main__':

    # configuration:
    num_agents = 2
    discount_factor = 0.99
    learning_rate = 0.1
    exploration_parameter = 0.2 

    # environment initialization
    env = gym.make(f"Foraging-8x8-{num_agents}p-1f-v3")
    obs = env.reset()

    # agent initialization (independent learning)
    agents = []
    for i in range(num_agents):
        agents.append(MinimaxQ(num_agents, discount_factor, learning_rate, exploration_parameter))

    max_steps = 1000
    for _ in range(max_steps):
        time.sleep(0.5)
        actions = env.action_space.sample()
        observations, rewards, done, truncated, info = env.step(actions)
            # observations : (array([2., 5., 2., 1., 0., 2., 6., 3., 1.], dtype=float32), array([2., 5., 2., 6., 3., 1., 1., 0., 2.], dtype=float32))
            # rewards : [0, 0]
            # done : False
            # truncated : False
            # info : {}  
        env.render()
          

