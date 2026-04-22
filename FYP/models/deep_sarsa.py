# Deep SARSA agent
# Implements the approach from:
# Mohamed and Ejbali's "Deep SARSA-based reinforcement learning approach for anomaly network intrusion detection system."

import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random


class QNetwork(nn.Module):
    def __init__(self, input_dim, n_actions, hidden_dims):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class DeepSARSA:
    def __init__(self, input_dim, n_actions, lr, gamma, epsilon,
                 hidden_dims, buffer_size=10000, batch_size=64,
                 epsilon_decay=0.995, epsilon_min=0.01):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        self.q_net = QNetwork(input_dim, n_actions, hidden_dims)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = deque(maxlen=buffer_size)

    # Epsilon-greedy action selection.
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            q_values = self.q_net(torch.FloatTensor(state))
        return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, next_action):
        self.replay_buffer.append((state, action, reward, next_state, next_action))

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, next_actions = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        next_actions = torch.LongTensor(next_actions)

        # Q(s, a) — Q-value
        q_pred = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # SARSA target: r + γ · Q(s', a')
        with torch.no_grad():
            q_next = self.q_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            q_target = rewards + self.gamma * q_next

        loss = self.loss_fn(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def predict(self, X):
        with torch.no_grad():
            q_values = self.q_net(torch.FloatTensor(X))
        return q_values.argmax(dim=1).numpy()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path, weights_only=True))
