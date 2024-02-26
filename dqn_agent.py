import random
import gymnasium as gym

from environment import SnakeEnv
from stable_baselines3 import DQN

env = SnakeEnv(render_mode='human')

model = DQN("MlpPolicy", env, verbose=1)

num_episodes = 10_000

# model.learn(total_timesteps=num_episodes)

obs = env.reset()
for episode in range(num_episodes):
    obs, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    print("score: {}".format(env.score))
