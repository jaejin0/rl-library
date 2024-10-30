from pettingzoo.atari import basketball_pong_v3

# Joint-Action Learning with Game Theory 

# JAL-GT have variations, such as Minimax-Q, Nash-Q, and CE-Q. This program used NashQ to converge to Nash equilibrium 
# For each state of the game, create a normal-form game with Q values.
# solution concepts:
# Minimax : solution concept for a two-agent zero-sum game. 
# Nash Equilibrium (NE) : solution concept for a any number of agents general-sum game.
# Correlated Equilibrium (CE) : NE with a countable probability space, as expected value can be greater than NE.

class MinimaxQ:
    def __init__():
        






if __name__ == '__main__':
    env = basketball_pong_v3.env(render_mode="human")
    env.reset()

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            # this is where you would insert your policy
            action = env.action_space(agent).sample()

        env.step(action)
    env.close()



