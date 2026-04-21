# models/deep_sarsa.py — Deep SARSA agent
#
# Implements the approach from:
#   Mohamed, S. and Ejbali, R. (2023). "Deep SARSA-based reinforcement learning
#   approach for anomaly network intrusion detection system."
#   International Journal of Information Security, 22(1), pp.235-247.
#
# Key design decisions matching their paper:
#   - Deep neural network approximates Q(s, a) for each action
#   - On-policy SARSA update: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') − Q(s,a)]
#     The critical SARSA property: a' is the ACTUAL next action taken by the
#     current policy, NOT the greedy max (that would be Q-learning / DQN).
#   - Experience replay buffer for scalable training (as introduced in their paper)
#   - Epsilon-greedy exploration with decay
#
# Extension from their paper: applied to CICIDS2017 under simulated concept drift
# (original paper used NSL-KDD and UNSW-NB15 datasets).

import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random


class QNetwork(nn.Module):
    """
    Feedforward network that approximates Q(s, a) for all actions simultaneously.

    Input:  feature vector (network flow state)
    Output: one Q-value per possible action

    Architecture: configurable hidden layers with ReLU activations,
    following standard deep RL practice.
    """

    def __init__(self, input_dim, n_actions, hidden_dims):
        """
        Args:
            input_dim:   number of input features (78 for CICIDS2017 after preprocessing)
            n_actions:   number of output actions (2 for binary IDS: BENIGN / ATTACK)
            hidden_dims: list of hidden layer sizes, e.g. [128, 64]
        """
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
    """
    Deep SARSA agent with experience replay (Mohamed & Ejbali, 2023).

    Training flow per timestep:
        1. Observe state s
        2. Select action a via epsilon-greedy  ← on-policy: policy chooses a
        3. Execute a, observe reward r and next state s'
        4. Select next action a' via epsilon-greedy  ← on-policy: same policy chooses a'
        5. Store (s, a, r, s', a') in replay buffer
        6. Sample a random mini-batch and perform one gradient update using SARSA target

    What makes this SARSA (not DQN):
        The update uses Q(s', a') where a' was actually selected by the current policy.
        DQN would use max_a' Q(s', a'), which is off-policy.
    """

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

        # Replay buffer stores (s, a, r, s', a') — the full SARSA quintuple
        self.replay_buffer = deque(maxlen=buffer_size)

    def select_action(self, state):
        """
        Epsilon-greedy action selection.

        Args:
            state: np.ndarray of shape (n_features,)
        Returns:
            action: int (0 or 1)
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            q_values = self.q_net(torch.FloatTensor(state))
        return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, next_action):
        """Push one (s, a, r, s', a') SARSA transition into the replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, next_action))

    def update(self):
        """
        Sample a mini-batch and perform one SARSA gradient update.

        SARSA target: r + γ · Q(s', a')
        Uses a' (the action actually taken), NOT max over all actions.
        Skips if buffer doesn't have enough samples yet.

        Returns:
            loss value (float) or None if skipped
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, next_actions = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        next_actions = torch.LongTensor(next_actions)

        # Q(s, a) — Q-value of the action that was actually taken
        q_pred = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # SARSA target: r + γ · Q(s', a')
        # next_actions are the actual actions taken in s', not the greedy best
        with torch.no_grad():
            q_next = self.q_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            q_target = rewards + self.gamma * q_next

        loss = self.loss_fn(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def predict(self, X):
        """
        Greedy (no exploration) predictions on a batch of states.
        Used during evaluation — epsilon is ignored.

        Args:
            X: np.ndarray of shape (n_samples, n_features)
        Returns:
            predictions: np.ndarray of shape (n_samples,)
        """
        with torch.no_grad():
            q_values = self.q_net(torch.FloatTensor(X))
        return q_values.argmax(dim=1).numpy()

    def decay_epsilon(self):
        """Reduce exploration rate. Call once at the end of each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        """Save model weights to disk."""
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        """Load model weights from disk."""
        self.q_net.load_state_dict(torch.load(path, weights_only=True))
