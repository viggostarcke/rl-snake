import argparse

from environment import SnakeEnv
from stable_baselines3 import DQN

parser = argparse.ArgumentParser(description='Learning')
parser.add_argument('--learn', default=False, action='store_true')
parser.add_argument('-r', default=False, action='store_true')


env = SnakeEnv()

if parser.parse_args().r:
    env.render_mode = "human"  # render every game

num_episodes = 300_000
learn = False
learn = parser.parse_args().learn

if learn:
    model = DQN("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=num_episodes)
    model.save("dqn_agent")
else:
    model = DQN.load("dqn_agent")
    obs = env.reset()
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        print("score: {}".format(env.score))
