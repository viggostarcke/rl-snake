import random
import gymnasium as gym

from environment import SnakeEnv
from stable_baselines3 import DQN

env = SnakeEnv()
env.render_mode = "human"  # render every game

num_episodes = 100_000
learn = True

if learn:
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=num_episodes)
    model.save("dqn_snake")
else:
    # Does not work properly: Snake gets stuck in loops
    model = DQN.load("dqn_snake")
    obs = env.reset()
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        print("score: {}".format(env.score))
