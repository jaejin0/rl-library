import torch
from vmas import make_env
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy, RandomPolicy

#def make_env(
#    scenario: Union[str, BaseScenario],
#    num_envs: int,
#    device: DEVICE_TYPING = "cpu",
#    continuous_actions: bool = True,
#    wrapper: Optional[Union[Wrapper, str]] = None,
#    max_steps: Optional[int] = None,
#    seed: Optional[int] = None,
#    dict_spaces: bool = False,
#    multidiscrete_actions: bool = False,
#    clamp_actions: bool = False,
#    grad_enabled: bool = False,
#    terminated_truncated: bool = False,
#    wrapper_kwargs: Optional[dict] = None,
#    **kwargs,
# ):

env = make_env(
        scenario="transport", # can be scenario name or BaseScenario class
        num_envs=1,
        device="cpu", # Or "cuda" for GPU
        continuous_actions=False,
        wrapper=None,  # One of: None, "rllib", "gym", "gymnasium", "gymnasium_vec"
        max_steps=None, # Defines the horizon. None is infinite horizon.
        seed=None, # Seed of the environment
        dict_spaces=False,
        multidiscrete_actions=False,
        grad_enabled=False,
        terminated_truncated=False,
    )

obs = env.reset()

for _ in range(1000):
    
    print("observation is : ", obs)
    print("observation space is : ", env.observation_space)
    print("action space is : ", env.action_space)
    actions = env.get_random_actions()
    print("action is : ", actions)
    obs, rews, dones, info = env.step(actions)
    rewards = torch.stack(rews, dim=1)
    global_reward = rewards.mean(dim=1)
    mean_global_reward = global_reward.mean(dim=0)
    env.render(
        mode="human",
        agent_index_focus=None,
    )
