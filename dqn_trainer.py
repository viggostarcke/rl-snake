import pygame

from dqn_agent import DQNAgent
from environment import SnakeEnv

env = SnakeEnv()
env.render_mode = "human"
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNAgent(state_dim, action_dim)

num_episodes = 10

for episode in range(num_episodes):
    state, _ = env.reset()
    reward = 0

    while True:
        env.render()

        if pygame.get_init() and hasattr(env, 'window') and env.window is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    pygame.quit()
                    exit()
        else:
            print("env render problem")

        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)

        agent.store_transition(state, action, reward, next_state, done)
        agent.experience_replay()

        state = next_state
        reward += reward

        if done:
            break

    print(f"Episode {episode+1}, Reward: {reward}")

    if episode % 10 == 0:
        agent.update_target_net()
