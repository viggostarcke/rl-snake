import argparse

from environment import SnakeEnv
from stable_baselines3 import DQN

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
    model = DQN(
        "MlpPolicy",
        env,
        gamma=0.99,
        exploration_final_eps=0.01
    )
    # model = DQN.load("dqn_agent")
    # model.set_env(env)
    model.learn(total_timesteps=num_timesteps)
    model.save("dqn_agent")
if test:
    model = DQN.load("dqn_agent")
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
