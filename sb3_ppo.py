import gymnasium as gym

from stable_baselines3 import PPO

env = gym.make("CartPole-v1", render_mode='human')

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

print("HEHEHEH")
vec_env = model.get_env()
obs = vec_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()

env.close()
