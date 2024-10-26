import torch
import numpy as np
from vmas import make_env

# Temporal-Difference (TD) Learning is a family of RL algorithm that learn optimal policies and value functions based on data collected via environment interations.
# For an example of TD Learning, V(s^t) <- V(s^t) + alpha(X - V(s^t)), where alpha is learning rate and X is the update target.
# And for action value function, Q(s^t, a^t) <- Q(s^t, a^t) + alpha(X - Q(s^t, a^t)).
# Central Q-learning is a TD algorithm for multi-agents which uses Bellman equation to update its value function estimates
# TD learning, off-policy
class CentralQLearning():
    # 5-tuple (S, A, gamma, alpha, epsilon)
    def __init__(self, state_space, action_space, discount_factor, learning_rate, exploration_parameter):
        self.state_space = state_space # 1-D array of state space
        self.action_space = action_space # 1-D array of action space
        self.discount_factor = discount_factor # <= 1 (float)
        self.learning_rate = learning_rate # (float)
        self.exploration_parameter = exploration_parameter # probability of choosing random action <= 1 (float)
    
        self.action_value_table = np.zeros([len(state_space), len(action_space)])
        self.prev_state = None
        self.prev_action = None

    def policy(self, state):
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
            scenario="waterfall",
            num_envs=32,
            device="cpu",
            continuous_actions=True,
            wrapper=None,
            max_steps=None,
            seed=None,
            dict_spaces=False,
            grad_enabled=False,
            terminated_truncated=False,
            n_agents = 10 
        )
    env.seed(0)
    n_agents = 10
    obs = env.reset()
    history = []
    for _ in range(1000):
        actions = []
        for i in range(n_agents):
            obs_agent = obs[i]
            action_agent = torch.clamp(
                obs_agent[:, -2:],
                min=-env.agents[i].u_range,
                max=env.agents[i].u_range,
            
            )
            actions.append(action_agent)

        obs, new_rews, _, _ = env.step(actions)

        frame = env.render(mode='rgb_array',
                           env_index=0,
                           agent_index_focus=None)
        history.append(frame)
