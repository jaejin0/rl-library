from numpy.random import choice 
from mdp import MDP

# Dynamic Programming (DP) assumes complete knowledge of MDP, including the reward function and transition function. We use Bellman equation to find optimal value functions.
# Value Iteration is a DP algorithm which uses Bellman equation to find optimal policy.
# Policy Iteration returns policy when policy is stable, but may not be optimal.
class ValueIteration:
    # n-tuple (S, A, R, T, gamma) 
    def __init__(self, state_space, action_space, reward_function, transition_function, discount_factor, iteration_limit):
        self.state_space = state_space # 1-D array of state space
        self.action_space = action_space # 1-D array of action space
        self.reward_function = reward_function # function(state, action, next_state) -> float 
        self.transition_function = transition_function # function(state, action, next_state) -> float
        self.discount_factor = discount_factor # <= 1 (float)
        self.iteration_limit = iteration_limit # depth of bellman equaition (int)
        
    def state_value_policy(self, state) -> float:
        # optimal greedy policy
        max_action, max_value = choice(action_space), 0
        for a in action_space:
            sum_over_state_space = 0
            for next_s in self.state_space:
                sum_over_state_space += self.transition_function(state, a, next_s) * (reward_function(state, a, next_s) + (self.discount_factor * self.state_value_function(next_s, self.iteration_limit - 1)))
            
            if sum_over_state_space > max_value:
                max_value = sum_over_state_space
                max_action = a

        return max_action

    def action_value_policy(self, state) -> float:
        # optimal greedy policy
        max_action, max_value = choice(action_space), 0
        for a in action_space:
            action_value = self.action_value_function(state, a)
            if action_value > max_value:
                max_value = action_value
                max_action = a
        
        return max_action

    def state_value_function(self, state, iteration_limit = None) -> float:        
        iteration_limit = self.iteration_limit if iteration_limit == None else iteration_limit
        if iteration_limit <= 0:
            return 0
        sum_over_action_space = 0
        for a in self.action_space:
            sum_over_state_space = 0
            for next_s in self.state_space:
                sum_over_state_space += self.transition_function(state, a, next_s) * (reward_function(state, a, next_s) + (self.discount_factor * self.state_value_function(next_s, iteration_limit - 1)))
            sum_over_action_space += sum_over_state_space
        
        return sum_over_action_space

    def action_value_function(self, state, action, iteration_limit = None) -> float:
        iteration_limit = self.iteration_limit if iteration_limit == None else iteration_limit
        if iteration_limit <= 0:
            return 0
        sum_over_state_space = 0
        for next_s in self.state_space:
            sum_over_action_space = 0
            for next_a in self.action_space:
                sum_over_action_space += self.discount_factor * self.action_value_function(next_s, next_a, iteration_limit - 1)
            sum_over_state_space += self.transition_function(state, action, next_s) * (reward_function(state, action, next_s) + sum_over_action_space)
        
        return sum_over_state_space


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

    print(state_space, initial_state_distribution) 
    mdp = MDP(state_space, terminal_state_space, action_space, reward_function, transition_function, initial_state_distribution)
    print(mdp.current_state)

    discount_factor = 0.5
    iteration_limit = 4
    policy = ValueIteration(state_space, action_space, reward_function, transition_function, discount_factor, iteration_limit)

    while True:
        current_state = mdp.current_state
        current_reward = mdp.current_reward
        print(f"Current state is  : {current_state}")
        print(f"Current reward is : {current_reward}")

        print(f"Current state value is : {policy.state_value_function(current_state)}")
        action = policy.state_value_policy(current_state)
        print(f"Action chosen by state_value_policy agent is : {action}")
        
        print(f"Current action value is : {policy.action_value_function(current_state, action)}")
        action = policy.action_value_policy(current_state)
        print(f"Action chosen by action_value_policy agent is : {action}")

        foo = input(f"press any key to forward")
        mdp.forward(action)
