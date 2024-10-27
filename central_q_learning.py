import torch
from vmas import make_env
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy, RandomPolicy


# Temporal-Difference (TD) Learning is a family of RL algorithm that learn optimal policies and value functions based on data collected via environment interations.
# For an example of TD Learning, V(s^t) <- V(s^t) + alpha(X - V(s^t)), where alpha is learning rate and X is the update target.
# And for action value function, Q(s^t, a^t) <- Q(s^t, a^t) + alpha(X - Q(s^t, a^t)).
# Central Q-learning is a TD algorithm for multi-agents which uses Bellman equation to update its value function estimates
# TD learning, off-policy

# This modification uses POMDP, which uses observation, as this utilize VMAS, a MARL simulation.
# POMDP : 7-tuple (S, A, T, R, omega, O, gamma)
class CentralQLearning():
    # 5-tuple (O, A, gamma, alpha, epsilon)
    def __init__(self, observation_space, action_space, discount_factor, learning_rate, exploration_parameter):
        self.observation_space = observation_space # (list of tensor)
        self.action_space = action_space # (list of tensor)
        self.discount_factor = discount_factor # <= 1 (float)
        self.learning_rate = learning_rate # (float)
        self.exploration_parameter = exploration_parameter # probability of choosing random action <= 1 (float)
    
        self.action_value_table = np.zeros([len(state_space), len(action_space)])
        self.prev_state = None
        self.prev_action = None

    def policy(self, state):
        # discretize state
        choose_random_action = np.random.binomial(1, self.exploration_parameter)
        if choose_random_action:
            action = np.random.choice(self.action_space)
        else:
            action, max_value = np.random.choice(self.action_space), 0 
            state_index = self.state_space.index(state)
            for a_index in range(len(action_space)):
                action_value = self.action_value_table[state_index][a_index]
                if action_value > max_value:
                    max_value = action_value
                    action = self.action_space[a_index]
        
        self.prev_state = state
        self.prev_action = action

        return action

    def learn_action_value_function(self, state, reward):
        max_value = 0
        for a in action_space:
            max_value = max(max_value, self.action_value_table[self.state_space.index(state)][self.action_space.index(a)])
        self.action_value_table[self.state_space.index(self.prev_state)][self.action_space.index(self.prev_action)] += \
            self.learning_rate * (reward + (self.discount_factor * max_value) -  
                                  self.action_value_table[self.state_space.index(self.prev_state)][self.action_space.index(self.prev_action)])


if __name__ == "__main__":
    
    env = make_env(
            scenario="wheel", # can be scenario name or BaseScenario class
            num_envs=32,
            device="cpu", # Or "cuda" for GPU
            continuous_actions=True,
            wrapper=None,  # One of: None, "rllib", "gym", "gymnasium", "gymnasium_vec"
            max_steps=1000, # Defines the horizon. None is infinite horizon.
            seed=None, # Seed of the environment
            dict_spaces=False,
            grad_enabled=False,
            terminated_truncated=False,
            n_agents = 2,
    )
    policy = RandomPolicy(continuous_action=True)

    obs = env.reset()

    for _ in range(1000):
        actions = [None] * len(obs)
        for i in range(len(obs)):
            actions[i] = policy.compute_action(obs[i], u_range=env.agents[i].u_range)
        print(f"This is the action\n: {env.action_space}")
        print(f"This is the observation\n: {obs}")
        obs, rews, dones, info = env.step(actions)
        rewards = torch.stack(rews, dim=1)
        global_reward = rewards.mean(dim=1)
        mean_global_reward = global_reward.mean(dim=0)
        env.render(
            mode="human",
            agent_index_focus=None,
        )
