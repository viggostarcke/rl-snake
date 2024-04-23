import argparse
import wandb

from environment import SnakeEnv
from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback

parser = argparse.ArgumentParser(description='Learning')
parser.add_argument('--learn', default=False, action='store_true')
parser.add_argument('--render', default=False, action='store_true')
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('-l', default=False, action='store_true')
parser.add_argument('-r', default=False, action='store_true')
parser.add_argument('-t', default=False, action='store_true')

env = SnakeEnv()

if parser.parse_args().r or parser.parse_args().render:
    env.render_mode = "human"  # render every game
learn = True if parser.parse_args().learn or parser.parse_args().l else False
test = True if parser.parse_args().test or parser.parse_args().t else False

num_timesteps = 1_000_000
num_games = 100

if learn:
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        n_epochs=10,
        verbose=1,
        tensorboard_log="./ppo_snake_tensorboard/"
    )
    # model = PPO.load("ppo_agent")
    # model.set_env(env)
    model.learn(total_timesteps=num_timesteps)
    model.save("ppo_agent")
if test:
    model = PPO.load("ppo_agent")
    obs = env.reset()
    total_score = 0
    for episode in range(num_games):
        obs, info = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        print("score: {}".format(env.score))
        total_score += env.score
    print("avg. score: {}".format(total_score / num_games))
