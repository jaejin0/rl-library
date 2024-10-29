import torch
from vmas import make_env
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy, RandomPolicy
import numpy as np
from itertools import combinations

# Temporal-Difference (TD) Learning is a family of RL algorithm that learn optimal policies and value functions based on data collected via environment interations.
# For an example of TD Learning, V(s^t) <- V(s^t) + alpha(X - V(s^t)), where alpha is learning rate and X is the update target.
# And for action value function, Q(s^t, a^t) <- Q(s^t, a^t) + alpha(X - Q(s^t, a^t)).
# Independent Q-learning is a TD algorithm for multi-agents which uses Bellman equation to update its value function estimates
# TD learning, off-policy

# This modification uses POMDP, which uses observation, as this utilize VMAS, a MARL simulation. And, as the environment
# POMDP : 7-tuple (S, A, T, R, omega, O, gamma)
class CentralQLearning():
    # 6 inputs (N, O, A, gamma, alpha, epsilon)
    def __init__(self, joint_observation_space, joint_action_space, discount_factor, learning_rate, exploration_parameter):
        self.n_agents = len(joint_observation_space)
        self.joint_observation_space = joint_observation_space # (Tuple of Box)
        self.joint_action_space = joint_action_space # (Tuple of Box)
        self.discount_factor = discount_factor # <= 1 (float)
        self.learning_rate = learning_rate # (float)
        self.exploration_parameter = exploration_parameter # probability of choosing random action <= 1 (float)
   
        # discretized observation and action, by creating integer list and scale down to float type
        OBS_RANGE, OBS_STEP = [-10, 10], 1
        ACTION_RANGE, ACTION_STEP = [int(self.joint_action_space[0].low[0]) * 10, int(self.joint_action_space[0].high[0]) * 10], 1 
        self.discrete_observation_space = [i/10 for i in range(OBS_RANGE[0], OBS_RANGE[1], OBS_STEP)]
        self.discrete_action_space = [i/10 for i in range(ACTION_RANGE[0], ACTION_RANGE[1], ACTION_STEP)]
        
        self.joint_action_value_table = {} # key : Tuple(Tuple(discrete_joint_observation), Tuple(discrete_joint_action))
        
        self.prev_joint_observation = None
        self.prev_joint_action = None

    def policy(self, joint_observation):
        # discretize the joint observation and store as a Tuple
        discrete_joint_observation = ()
        for obs in joint_observation:
            discrete_observation = []
            for o in obs[0]:
                discrete_observation.append(min(self.discrete_observation_space, key=lambda x: abs(x - o)))
            discrete_joint_observation += tuple(discrete_observation)
        
        # explore
        discrete_joint_action = []
        for i in range(self.n_agents):
            discrete_joint_action.append(torch.from_numpy(np.expand_dims(np.random.choice(self.discrete_action_space, self.joint_action_space[0].shape[0]), 0)))
        discrete_joint_action = tuple(discrete_joint_action)
         
        choose_random_joint_action = np.random.binomial(1, self.exploration_parameter)
        if not choose_random_joint_action: # exploit based on exploration parameter
            max_val = 0
            for k in self.joint_action_value_table:
                if k[0] != discrete_joint_observation:
                    continue

                if self.joint_action_value_table[k] > max_val:
                    max_val = self.joint_action_value_table[k]
                    discrete_joint_action = k[1]

        self.prev_discrete_joint_observation = discrete_joint_observation
        self.prev_discrete_joint_action = discrete_joint_action

        return discrete_joint_action

    def learn_joint_action_value_function(self, joint_observation, joint_reward):
        # finding tuple of discretized joint observation
        discrete_joint_observation = ()
        for obs in joint_observation:
            discrete_observation = []
            for o in obs[0]:
                discrete_observation.append(min(self.discrete_observation_space, key=lambda x: abs(x - o)))
            discrete_joint_observation += tuple(discrete_observation)
        
        max_value = 0 
        for next_discrete_joint_a in combinations(self.discrete_action_space, self.n_agents):
            # search how to get certain number of every combination of a list
            
            max_value = max(max_value, self.joint_action_value_table.get((discrete_joint_observation, next_discrete_joint_a), 0))

        self.joint_action_value_table[(self.prev_discrete_joint_observation, self.prev_discrete_joint_action)] = \
                self.joint_action_value_table.get((self.prev_discrete_joint_observation, self.prev_discrete_joint_action), 0) + \
                self.learning_rate * (joint_reward + (self.discount_factor * max_value) - 
                                      self.joint_action_value_table.get((self.prev_discrete_joint_observation, self.prev_discrete_joint_action), 0))
  

if __name__ == "__main__":
    
    env = make_env(
            scenario="wheel", # can be scenario name or BaseScenario class
            num_envs=1,
            device="cpu", # Or "cuda" for GPU
            continuous_actions=True,
            wrapper=None,  # One of: None, "rllib", "gym", "gymnasium", "gymnasium_vec"
            max_steps=1000, # Defines the horizon. None is infinite horizon.
            seed=None, # Seed of the environment
            dict_spaces=False,
            grad_enabled=False,
            terminated_truncated=False,
            n_agents = 3,
    )
    
    discount_factor = 0.5
    learning_rate = 0.1
    exploration_parameter = 0.3

    agent = CentralQLearning(env.observation_space, env.action_space, discount_factor, learning_rate, exploration_parameter) 
    print(env.max_steps)
    episode = 100
    for ep in range(episode):
        print(f"EPISODE: {ep}")
        joint_observation = env.reset()
        for _ in range(env.max_steps):
            joint_action = agent.policy(joint_observation)
            print(f"This is the observation\n: {joint_observation}")
            print(f"This is the action\n: {joint_action}")
            joint_observation, joint_reward, dones, info = env.step(joint_action)
            print(f"This is reward: {joint_reward}")
            print(f"This is dones: {dones}")
            print(f"This is info: {info}")
            total_reward = torch.stack(joint_reward, dim=1).mean(dim=1).mean(dim=0)[0].item()
            print(f"total reward: {total_reward}")
            agent.learn_joint_action_value_function(joint_observation, total_reward)

            if dones[0].item() == True:
                joint_observation = env.reset()
            env.render(
                mode="human",
                agent_index_focus=None,
            )
            # input("press any key to continue")
