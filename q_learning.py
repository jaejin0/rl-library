import numpy as np
from mdp import MDP

# Temporal-Difference (TD) Learning is a family of RL algorithm that learn optimal policies and value functions based on data collected via environment interations.
# For an example of TD Learning, V(s^t) <- V(s^t) + alpha(X - V(s^t)), where alpha is learning rate and X is the update target.
# And for action value function, Q(s^t, a^t) <- Q(s^t, a^t) + alpha(X - Q(s^t, a^t)).
# Q-learning is a TD algorithm which uses Bellman equation to update its value function estimates
# TD learning, off-policy
class QLearning():
    # 5 inputs (S, A, gamma, alpha, epsilon)
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
    
    # MDP specification
    grid_height = 3
    grid_width = 4

    state_space = []
    for i in range(grid_height):
        for j in range(grid_width):
            state_space.append((i, j))
    terminal_state_space = [(grid_height - 1, grid_width - 1)]

    initial_state_distribution = []
    for i in range(grid_height):
        for j in range(grid_width):
            initial_state_distribution.append(0)

    initial_state_distribution[0] = 0.6
    initial_state_distribution[1] = 0.2
    initial_state_distribution[grid_width] = 0.2

    action_space = ['left', 'right', 'up', 'down', 'no-op'] 

    def reward_function(state, action, next_state):
        if next_state == (grid_height - 1, grid_width - 1): # goal
            return 1
        else:
            return 0

    def transition_function(state, action, next_state):
        match action:
            case 'left':
                if state[1] == 0 and state == next_state: # can't move to the left further
                    return 1
                if state[1] != 0 and state[1] - 1 == next_state[1] and state[0] == next_state[0]:
                    return 1
            case 'right':
                if state[1] == grid_width - 1 and state == next_state: # can't move to the right further
                    return 1
                if state[1] != grid_width - 1 and state[1] + 1 == next_state[1] and state[0] == next_state[0]:
                    return 1
            case 'up':
                if state[0] == 0 and state == next_state: # can't move up further
                   return 1
                if state[0] != 0 and state[0] - 1 == next_state[0] and state[1] == next_state[1]:
                   return 1
            case 'down':
                if state[0] == grid_height - 1 and state == next_state: # can't move down further
                    return 1
                if state[0] != grid_height - 1 and state[0] + 1 == next_state[0] and state[1] == next_state[1]:
                    return 1
            case 'no-op':
                if state == next_state:
                    return 1

        return 0
    
    mdp = MDP(state_space, terminal_state_space, action_space, reward_function, transition_function, initial_state_distribution)

    discount_factor = 0.5
    learning_rate = 0.1
    exploration_parameter = 0.5
    agent = QLearning(state_space, action_space, discount_factor, learning_rate, exploration_parameter)

    while True:
        print(f"{agent.action_value_table}")
        current_state = mdp.current_state
        current_reward = mdp.current_reward
        print(f"Current state is  : {current_state}")
        print(f"Current reward is : {current_reward}")

        action = agent.policy(current_state)
        print(f"Action chosen by state_value_policy agent is : {action}")

        # foo = input(f"press any key to forward")
        mdp.forward(action)

        next_state = mdp.current_state
        current_reward = mdp.current_reward
        agent.learn_action_value_function(next_state, current_reward)
