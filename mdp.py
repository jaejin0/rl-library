from numpy.random import choice

class MDP:
    # 5-tuple (S, A, R, T, mu)
    def __init__(self, state_space, action_space, reward_function, transition_function, initial_state_distribution):
        self.state_space = state_space # [s1, s2, ... ]
        self.action_space = action_space # [a1, a2, ... ]
        self.reward_function = reward_function # function(state, action, next_action) -> float
        self.transition_function = transition_function # function(state, action, next_action) -> float
        self.initial_state_distribution = initial_state_distribution # [pr1, pr2, ... ] # same size as the state_space, probabilties sums up to 1. 

    def start(self):
        self.current_state = choice(self.state_space, p=self.initial_state_distribution)

    def forward(self, action):
        # update position of state with the value of action
        next_state = choice(self.state_space, p=self.probability_of_next_state(self.current_state, action))
        reward = self.reward_function(self.current_state, action, next_state)
        self.current_state = next_state
        return reward

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

    state_space = [0, 1, 2, 3]
    action_space = ["left", "no-op", "right"]
    initial_state_distribution = [0, 0.2, 0.5, 0.3]

    mdp = MDP(state_space, action_space, reward_function, transition_function, initial_state_distribution)

    mdp.start()
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
        print(f"reward: {mdp.forward(user_input)}")
