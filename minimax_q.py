from pettingzoo.atari import basketball_pong_v3

# Joint-Action Learning with Game Theory 

# JAL-GT have variations, such as Minimax-Q, Nash-Q, and CE-Q. This program used NashQ to converge to Nash equilibrium 
# For each state of the game, create a normal-form game with Q values.
# solution concepts:
# Minimax : solution concept for a two-agent zero-sum game. 
# Nash Equilibrium (NE) : solution concept for a any number of agents general-sum game.
# Correlated Equilibrium (CE) : NE with a countable probability space, as expected value can be greater than NE.

# JAL-GT pseudocode
# // Algorithm controls agent i
# Initialize: Q_j(s,a) = 0 for all j in I and s in S, a in A
# Repeat for every episode:
# for t = 0, 1, 2, ... do
#     Observe current state s^t
#     With probability epsilon: choose random action a^t_i
#     Otherwise: solve Gamma_{s^t} to get policies (pi_1, ..., pi_n), then sample action a^t_i ~ pi_i
#     Observe joint action a^t = (a^t_1, ..., a^t_n), rewards r^t_1, ..., r^t_n, next state s^{t+1}
#     for all j in I do
#         Q_j(s^t, a^t) <- Q_j(s^t, a^t) + alpha[r^t_j + gamma * Value_j(Gamma_{s^{t+1}}) - Q_j(s^t, a^t)]

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



