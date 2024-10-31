from numpy.random import choice

class MDP:
    # 5-tuple (S, A, R, T, mu)
    def __init__(self, state_space, terminal_state_space, action_space, reward_function, transition_function, initial_state_distribution):
        self.state_space = state_space # 1-D array of state space [s1, s2, ... ]
        self.terminal_state_space = terminal_state_space # 1-D array of terminal state space # subset of state space
        self.action_space = action_space # 1-D array of action space [a1, a2, ... ]
        self.reward_function = reward_function # function(state, action, next_state) -> float
        self.transition_function = transition_function # function(state, action, next_state) -> float
        self.initial_state_distribution = initial_state_distribution # 1-D array of probability of state space [pr1, pr2, ... ] # same size as the state_space, probabilties sums up to 1. 
        
        # initial state
        self.current_reward = 0 # at t = 0, reward is 0
        self.reset()
    
    def reset(self):
        state_index = choice(len(self.state_space), 1, p=self.initial_state_distribution)[0]
        self.current_state = self.state_space[state_index]

    def forward(self, action):
        next_state_index = choice(len(self.state_space), 1, p=self.probability_of_next_state(self.current_state, action))[0]
        next_state = self.state_space[next_state_index]
        self.current_reward = self.reward_function(self.current_state, action, next_state)
        
        if next_state in self.terminal_state_space
            self.reset()
        else:
            self.current_state = next_state

    def probability_of_next_state(self, state, action):
        probability_of_next_state = []
        for next_state in self.state_space:
            probability_of_next_state.append(self.transition_function(state, action, next_state))
        return probability_of_next_state


# Reward : S x A x S -> real_number
def reward_function(state, action, next_state) -> float:
    return state * next_state

# Transition : S x A x S -> [0, 1]
def transition_function(state, action, next_state) -> float:
    # no randomness
    if action == "left":
        if state > 0:
            if state - 1 == next_state:
                return 1
        else:
            if state == next_state:
                return 1
    elif action == "right":
        if state < 3:
            if state + 1 == next_state:
                return 1
        else:
            if state == next_state:
                return 1
    else:
        if state == next_state:
            return 1
    return 0


if __name__ == "__main__":

    # 1-D game
    state_space = [0, 1, 2, 3]
    terminal_state_space = [0]
    action_space = ["left", "no-op", "right"]
    initial_state_distribution = [0, 0.2, 0.5, 0.3]

    mdp = MDP(state_space, terminal_state_space, action_space, reward_function, transition_function, initial_state_distribution)

    while True:
        print(f"You are at {mdp.current_state}")
        print(f"Choose from the following: {action_space}")
        user_input = input()
        match user_input:
            case "a":
                action = "left"
            case "d":
                action = "right"
            case _:
                action = "no-op"
        mdp.forward(action)
        print(f"reward: {mdp.current_reward}")
