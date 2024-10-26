import torch
from vmas import make_env
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy, RandomPolicy

env = make_env(
        scenario="transport", # can be scenario name or BaseScenario class
        num_envs=32,
        device="cpu", # Or "cuda" for GPU
        continuous_actions=True,
        wrapper=None,  # One of: None, "rllib", "gym", "gymnasium", "gymnasium_vec"
        max_steps=None, # Defines the horizon. None is infinite horizon.
        seed=None, # Seed of the environment
        dict_spaces=False,
        grad_enabled=False,
        terminated_truncated=False,
    )
policy = RandomPolicy(continuous_action=True)

obs = env.reset()

for _ in range(1000):
    actions = [None] * len(obs)
    for i in range(len(obs)):
        actions[i] = policy.compute_action(obs[i], u_range=env.agents[i].u_range)

    obs, rews, dones, info = env.step(actions)
    rewards = torch.stack(rews, dim=1)
    global_reward = rewards.mean(dim=1)
    mean_global_reward = global_reward.mean(dim=0)
    env.render(
        mode="human",
        agent_index_focus=None,
    )
