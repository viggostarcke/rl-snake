import argparse
import wandb

from environment import SnakeEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from wandb.integration.sb3 import WandbCallback

parser = argparse.ArgumentParser(description='Learning')
parser.add_argument('--learn', default=False, action='store_true')
parser.add_argument('-r', default=False, action='store_true')

env = SnakeEnv()

if parser.parse_args().r:
    env.render_mode = "human"  # render every game

num_timesteps = 10_000_000
learn = False
learn = parser.parse_args().learn

if learn:
    # model = PPO(
    #     "MlpPolicy",
    #     env,
    #     learning_rate=3e-4,
    #     n_steps=2048,
    #     batch_size=128,
    #     gamma=0.99,
    #     gae_lambda=0.95,
    #     clip_range=0.2,
    #     ent_coef=0.01,
    #     n_epochs=10,
    #     verbose=1,
    #     tensorboard_log="./ppo_snake_tensorboard/"
    # )
    model = PPO.load("ppo_agent")
    model.set_env(env)
    model.learn(total_timesteps=num_timesteps)
    model.save("ppo_agent")
else:
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
