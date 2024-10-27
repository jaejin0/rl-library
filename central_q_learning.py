import torch
from vmas import make_env
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy, RandomPolicy
import numpy as np

# Temporal-Difference (TD) Learning is a family of RL algorithm that learn optimal policies and value functions based on data collected via environment interations.
# For an example of TD Learning, V(s^t) <- V(s^t) + alpha(X - V(s^t)), where alpha is learning rate and X is the update target.
# And for action value function, Q(s^t, a^t) <- Q(s^t, a^t) + alpha(X - Q(s^t, a^t)).
# Central Q-learning is a TD algorithm for multi-agents which uses Bellman equation to update its value function estimates
# TD learning, off-policy

# This modification uses POMDP, which uses observation, as this utilize VMAS, a MARL simulation.
# POMDP : 7-tuple (S, A, T, R, omega, O, gamma)
class CentralQLearning():
    # 6-tuple (N, O, A, gamma, alpha, epsilon)
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
        
        print(self.joint_action_space)
        print(len(self.joint_observation_space) * self.joint_observation_space[0].shape[0])

        print(len(self.joint_action_space) * self.joint_action_space[0].shape[0])
        self.joint_action_value_table = {} # key : Tuple(Tuple(discrete_joint_observation), Tuple(discrete_joint_action))
        # self.action_value_table = np.zeros([len(self.observation_space) * self.observation_space[0].shape[0], 
        #                                     len(self.action_space) * self.action_space[0].shape[0]])
        self.prev_joint_observation = None
        self.prev_joint_action = None

    def policy(self, joint_observation):
        # discretize state
        
        # explore
        joint_action = []
        for i in range(self.n_agents):
            print("here is", self.joint_action_space[0].shape[0], self.joint_action_space[0].low, self.joint_action_space[0].high[0])
            joint_action.append(torch.clamp(torch.randn(env.num_envs, self.joint_action_space[0].shape[0]), 
                                                self.joint_action_space[0].low[0], self.joint_action_space[0].high[0]))

        
        choose_random_joint_action = np.random.binomial(1, self.exploration_parameter)
        if not choose_random_joint_action: # exploit
            discrete_joint_observation = ()
            for obs in joint_observation:
                discrete_observation = []
                for o in obs[0]:
                    discrete_observation.append(min(self.discrete_observation_space, key=lambda x: abs(x - o)))
                discrete_joint_observation += tuple(discrete_observation)
            
            max_val = 0
            for k in self.joint_action_value_table:
                if k[0] != discrete_joint_observation:
                    continue

                if self.joint_action_value_table[k] > max_val:
                    max_val = self.joint_action_value_table[k]
                    joint_action = k[1]
                    print("The exploit action is : ", joint_action)

        self.prev_joint_observation = joint_observation
        self.prev_joint_action = joint_action

        return joint_action

    def learn_action_value_function(self, observation, reward):
        max_value = 0
        for a in action_space:
            max_value = max(max_value, self.action_value_table[self.state_space.index(state)][self.action_space.index(a)])
        self.action_value_table[self.state_space.index(self.prev_state)][self.action_space.index(self.prev_action)] += \
            self.learning_rate * (reward + (self.discount_factor * max_value) -  
                                  self.action_value_table[self.state_space.index(self.prev_state)][self.action_space.index(self.prev_action)])

   
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

    print(env.get_observation_space) 
    agent = CentralQLearning(env.observation_space, env.action_space, discount_factor, learning_rate, exploration_parameter) 

    policy = RandomPolicy(continuous_action=True)
    obs = env.reset()

    for _ in range(1000):
#        actions = [None] * len(obs)
 #       for i in range(len(obs)):
  #          actions[i] = policy.compute_action(obs[i], u_range=env.agents[i].u_range)
        actions = agent.policy(obs)
        print(f"This is the action\n: {actions}")
        print(f"This is the observation\n: {obs}")
        obs, joint_reward, dones, info = env.step(actions)
        print(f"This is reward: {joint_reward}")
        print(f"This is dones: {dones}")
        print(f"This is info: {info}")
        # rewards = torch.stack(rews, dim=1)
        # global_reward = rewards.mean(dim=1)
        # mean_global_reward = global_reward.mean(dim=0)
        env.render(
            mode="human",
            agent_index_focus=None,
        )
        input("press any key to continue")
