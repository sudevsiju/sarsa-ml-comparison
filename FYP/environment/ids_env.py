# environment/ids_env.py
# Wraps the IDS dataset as a reinforcement learning environment.
# Network flow records are presented to the agent one at a time as state
# observations, following Mohamed & Ejbali (2023). One full pass = one episode.

import numpy as np


class IDSEnvironment:
    """
    Gym-style environment where each timestep presents one network flow record.

    The agent receives the flow's feature vector as its state, selects a
    classification action, and receives a reward reflecting both correctness
    and the real-world cost of each error type.

    Reward schedule:
    +1.0  correct classification (any class)
    -1.0  missed attack — predicted BENIGN but sample was an attack (false negative)
    -1.0  false alarm — predicted an attack but sample was BENIGN (false positive)
    -0.8  wrong attack class — flagged as malicious but assigned the wrong category.
          Less severe than a missed attack because the traffic was still flagged.
    """

    def __init__(self, X, y):
        """
        Args:
            X: np.ndarray (n_samples, n_features) — scaled feature vectors
            y: np.ndarray (n_samples,) — class labels (0=BENIGN, 1=DoS, 2=DDoS, ...)
        """
        self.X         = X
        self.y         = y
        self.n_samples = len(X)
        self.idx       = 0

    def reset(self):
        """Reset to the first sample and return its feature vector."""
        self.idx = 0
        return self.X[0]

    def step(self, action):
        """
        Submit the agent's classification decision for the current sample.

        Args:
            action: int — predicted class

        Returns:
            next_state: feature vector of the next sample, or None at end of episode
            reward:     float — see class docstring for reward schedule
            done:       True when all samples have been processed
        """
        true_label = int(self.y[self.idx])
        action     = int(action)

        if action == true_label:
            reward = 1.0    # correct prediction
        elif true_label != 0 and action == 0:
            reward = -1.0   # missed attack (false negative)
        elif true_label == 0 and action != 0:
            reward = -1.0   # false alarm (false positive)
        else:
            reward = -0.8   # correct that it is an attack, but wrong attack class

        self.idx += 1
        done = self.idx >= self.n_samples

        next_state = None if done else self.X[self.idx]
        return next_state, reward, done

    def __len__(self):
        return self.n_samples
