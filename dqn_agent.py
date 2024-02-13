import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

from collections import deque
from dqn_model import DQN


class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=1e-4,
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay=0.995, memory_size=10000, batch_size=1000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def choose_action(self, state, explore=True):
        if explore and np.random.rand() <= self.epsilon:  # exploration
            return random.randrange(self.action_dim)
        else:  # exploitation
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.policy_net.eval()
            with torch.no_grad():
                action_values = self.policy_net(state)
            self.policy_net.train()
            return np.argmax(action_values.cpu().data.numpy())

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = random.sample(self.memory, self.batch_size)
        batch = list(zip(*transitions))

        states = np.array([s for s in batch[0]])
        next_states = np.array([s for s in batch[3]])

        states = torch.tensor(states, dtype=torch.float).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)

        # states = torch.stack([torch.tensor(s, dtype=torch.float).to(self.device) for s in batch[0]])
        # states = torch.tensor(np.vstack(batch[0]), dtype=torch.float).to(self.device)
        # next_states = torch.stack([torch.tensor(s, dtype=torch.float).to(self.device) for s in batch[3]])
        # next_states = torch.tensor(np.vstack(batch[3]), dtype=torch.float).to(self.device)

        actions = torch.tensor(batch[1], dtype=torch.long).to(self.device).view(-1, 1)
        rewards = torch.tensor(batch[2], dtype=torch.float).to(self.device).view(-1, 1)
        dones = torch.tensor(batch[4], dtype=torch.float).to(self.device).view(-1, 1)

        # compute q values for current states
        state_action_values = self.policy_net(states).gather(1, actions)
        # compute V(s{t01}) for all next states
        next_state_values = self.target_net(next_states).max(1)[0].detach()
        # compute the expected q values
        expected_state_action_values = (next_state_values * self.gamma * (1 - dones)) + rewards
        # compute huber loss
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
