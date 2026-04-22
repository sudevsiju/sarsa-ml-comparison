# Wraps the IDS dataset as a reinforcement learning environment.
# Network flow records are presented to the agent one at a time as state observations, following Mohamed & Ejbali (2023). One full pass = one episode.

import numpy as np


class IDSEnvironment: 
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.n_samples = len(X)
        self.idx = 0

    def reset(self):
        # Reset to the first sample and return its feature vector.
        self.idx = 0
        return self.X[0]

    def step(self, action):
        # Submit the agent's classification decision for the current sample.    
        true_label = int(self.y[self.idx])
        action = int(action)

        if action == true_label:
            reward = 1.0 # correct prediction
        elif true_label != 0 and action == 0:
            reward = -1.0 # missed attack (false negative)
        elif true_label == 0 and action != 0:
            reward = -1.0 # false alarm (false positive)
        else:
            reward = -0.8 # correct that it is an attack, but wrong attack class

        self.idx += 1
        done = self.idx >= self.n_samples

        next_state = None if done else self.X[self.idx]
        return next_state, reward, done

    def __len__(self):
        return self.n_samples
