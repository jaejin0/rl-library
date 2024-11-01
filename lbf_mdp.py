import gymnasium as gym
import lbforaging
import time

# Level-based Foraging (LBF): a multi-agent environment for RL
env = gym.make("Foraging-8x8-2p-1f-v3")
obs = env.reset()

max_steps = 1000
for _ in range(max_steps):
    time.sleep(0.5)
    actions = env.action_space.sample()
    print(env.step(actions))
    env.render()

