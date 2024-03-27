import argparse
import wandb

from environment import SnakeEnv
from stable_baselines3 import DQN
from wandb.integration.sb3 import WandbCallback

parser = argparse.ArgumentParser(description='Learning')
parser.add_argument('--learn', default=False, action='store_true')
parser.add_argument('-r', default=False, action='store_true')

# wandb.init(
#     project="SnakeEnv",
#     entity="visttt",
#     config={
#         "policy": "MultiInputPolicy",
#         "total_timesteps": 300_000,
#         "env_name": "SnakeEnv"
#     },
#     sync_tensorboard=True,
# )
#
# config = wandb.config

env = SnakeEnv()

if parser.parse_args().r:
    env.render_mode = "human"  # render every game

num_episodes = 200_000
learn = False
learn = parser.parse_args().learn

if learn:
    # model = DQN(
    #     "MultiInputPolicy",
    #     env,
    #     learning_rate=5e-4,
    #     buffer_size=50000,
    #     batch_size=32,
    #     gamma=0.99,
    #     exploration_initial_eps=1.0,
    #     exploration_final_eps=0.01,
    #     exploration_fraction=0.1,
    #     target_update_interval=1000,
    #     learning_starts=1000,
    #     train_freq=4,
    #     gradient_steps=1,
    #     verbose=1,
    #     tensorboard_log="./dqn_snake_tensorboard/"
    # )
    model = DQN.load("dqn_agent")
    model.set_env(env)
    model.learn(total_timesteps=num_episodes)
    model.save("dqn_agent")

    # model = DQN("MultiInputPolicy", env, verbose=1)  # learning_rate=1e-4, gamma=0.99, buffer_size=1_000_000, batch_size=128
    # model.learn(
    #     total_timesteps=num_episodes,
    #     callback=WandbCallback(
    #         gradient_save_freq=100,
    #         model_save_path=f"model/{run.id}",
    #         verbose=2
    #     )
    # )
    # model.save("dqn_agent")

    # model = DQN("MultiInputPolicy", env, verbose=1, tensorboard_log=f"runs/dqn")
    # for episode in range(num_episodes):
    #     obs, info = env.reset()
    #     done = False
    #     total_rewards = 0
    #     steps = 0
    #
    #     while not done:
    #         action, _states = model.predict(obs, deterministic=True)
    #         obs, rewards, terminated, truncated, info = env.step(action)
    #         done = terminated or truncated
    #
    #         # Accumulate rewards and steps
    #         total_rewards += rewards
    #         steps += 1
    #
    #         # Log steps or custom metrics if needed
    #         # wandb.log({'step_reward': rewards})
    #
    #     # Log episode metrics
    #     wandb.log({'episode_reward': total_rewards, 'episode_length': steps})
    #
    #     # Optional: Save model periodically
    #     if episode % 10000 == 0:
    #         model_path = f"dqn_agent_{episode}"
    #         model.save(model_path)
    #         wandb.save(model_path)
    #
    #     # Final model save
    # model.save("dqn_agent")
    # wandb.save("dqn_agent")
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
