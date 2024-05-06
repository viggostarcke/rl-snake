import argparse

from environment import SnakeEnv
from stable_baselines3 import PPO

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
num_games = 1000

if learn:
    model = PPO(
        "MultiInputPolicy",
        env,
        gamma=0.99,
        ent_coef=0.01
    )
    # model = PPO.load("ppo_agent_R3_10M")
    # model.set_env(env)
    model.learn(total_timesteps=num_timesteps)
    model.save("ppo_agent_R2_1M")
if test:
    model = PPO.load("ppo_agent_R2_1M")
    obs = env.reset()
    total_score = 0
    high_score = 0
    for episode in range(num_games):
        obs, info = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        print("score: {}".format(env.score))
        total_score += env.score
        if env.score > high_score:
            high_score = env.score
    print("avg. score: {}".format(total_score / num_games) +
          ", high score: {}".format(high_score) + ", loop incidents: {}".format(env.loop_counter))
