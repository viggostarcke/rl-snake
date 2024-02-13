from dqn_agent import DQNAgent
from environment import SnakeEnv

env = SnakeEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNAgent(state_dim, action_dim)

num_episodes = 1000

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    while True:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)

        agent.store_transition(state, action, reward, next_state, done)
        agent.experience_replay()

        state = next_state
        total_reward += reward

        if done:
            break

    print(f"Episode {episode+1}, Total reward: {total_reward}")

    if episode % 10 == 0:
        agent.update_target_net()
