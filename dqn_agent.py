import random
import gymnasium as gym

from environment import SnakeEnv
from stable_baselines3 import DQN

env = SnakeEnv()

model = DQN("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=10000)
