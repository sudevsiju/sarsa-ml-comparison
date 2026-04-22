# Deep Q-Network agent (off-policy RL baseline)
# Used as the RL comparison against Deep SARSA.
# Reference: Alavizadeh et al. (2022). "Deep Q-learning based reinforcement learning approach for network intrusion detection."

import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random

from models.deep_sarsa import QNetwork  # same architecture, reuse


class DQN:

    def __init__(self, input_dim, n_actions, lr, gamma, epsilon,
                 hidden_dims, buffer_size=10000, batch_size=64,
                 target_update_freq=10, epsilon_decay=0.995, epsilon_min=0.01):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_count = 0

        self.online_net = QNetwork(input_dim, n_actions, hidden_dims)
        self.target_net = QNetwork(input_dim, n_actions, hidden_dims)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = deque(maxlen=buffer_size)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            q = self.online_net(torch.FloatTensor(state))
        return q.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, float(done)))

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        q_pred = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next   = self.target_net(next_states).max(dim=1)[0]
            q_target = rewards + self.gamma * q_next * (1.0 - dones)

        loss = self.loss_fn(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.sync_target_network()

        return loss.item()

    def sync_target_network(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def predict(self, X):
        with torch.no_grad():
            q = self.online_net(torch.FloatTensor(X))
        return q.argmax(dim=1).numpy()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save(self.online_net.state_dict(), path)

    def load(self, path):
        self.online_net.load_state_dict(torch.load(path, weights_only=True))
        self.target_net.load_state_dict(self.online_net.state_dict())
