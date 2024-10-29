import gymnasium as gym
from q_learning import QLearning

# Gymnasium gives observation space, not state space
env = gym.make("CartPole-v1", render_mode="human")
# observation_space = env.observation_space
# action_space = env.action_space
# discount_factor = 0.5
# learning_rate = 0.1
# exploration_parameter = 0.5

# agent = QLearning(observation_space, action_space, discount_factor, learning_rate, exploration_parameter)



observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    # action = agent.policy(observation)
    observaiton, reward, terminated, truncated, info = env.step(action)

    # agent.learn_action_value_function(observation, reward) 

    if terminated or truncated:
        observation, info = env.reset()

env.close()
