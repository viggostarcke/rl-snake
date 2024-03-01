import argparse

from environment import SnakeEnv
from stable_baselines3 import PPO

parser = argparse.ArgumentParser(description='Learning')
parser.add_argument('--learn', default=False, action='store_true')
parser.add_argument('-r', default=False, action='store_true')


env = SnakeEnv()

if parser.parse_args().r:
    env.render_mode = "human"  # render every game

num_timesteps = 500_000
learn = False
learn = parser.parse_args().learn

if learn:
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=num_timesteps)
    model.save("ppo_agent")
else:
    # Does not work properly: Snake gets stuck in loops
    model = PPO.load("ppo_agent")
    obs = env.reset()
    for episode in range(num_timesteps):
        obs, info = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        print("score: {}".format(env.score))
