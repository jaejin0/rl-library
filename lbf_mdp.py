import gymnasium as gym
import lbforaging
import time

# Level-based Foraging (LBF): a multi-agent environment for RL
env = gym.make("Foraging-8x8-2p-1f-v3")
observations, info = env.reset()

max_steps = 1000
for _ in range(max_steps):
    time.sleep(0.5)
    actions = env.action_space.sample()
    observations, rewards, done, truncated, info = env.step(actions)
        # observations : (array([2., 5., 2., 1., 0., 2., 6., 3., 1.], dtype=float32), array([2., 5., 2., 6., 3., 1., 1., 0., 2.], dtype=float32))
        # rewards : [0, 0]
        # done : False
        # truncated : False
        # info : {}  
    env.render()

