import gymnasium as gym

from environment import SnakeEnv
from stable_baselines3 import PPO

env = SnakeEnv()

model = PPO("MlpPolicy", env, verbose=1)

num_episodes = 10_000

model.learn(total_timesteps=num_episodes)
