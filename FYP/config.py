# config.py — all hyperparameters and project-wide settings in one place

import os

# ── Paths ──────────────────────────────────────────────────────────────────
# Points directly at the kagglehub download cache — no copying needed locally.
# On Google Colab, change this to wherever you upload the CSVs on Drive.
DATA_DIR = r"C:\Users\sudev\.cache\kagglehub\datasets\chethuhn\network-intrusion-dataset\versions\1"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "cache")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ── Preprocessing ──────────────────────────────────────────────────────────
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ── Concept drift scenario (3-stage, multiclass) ──────────────────────────
#
# Labels:  0=BENIGN  1=DoS  2=DDoS  3=BruteForce  4=Other
#
# T1 — No drift (training distribution):
#   BENIGN + DoS only.  Models NEVER see DDoS or BruteForce here.
#
# T2 — Mild drift:
#   BENIGN + DoS + BruteForce.  BruteForce is a new class — mild novelty.
#
# T3 — Severe drift:
#   BENIGN + DDoS ONLY.  DDoS is completely unseen by any trained model.
#   Static models (RF, DQN) must mis-classify DDoS; SARSA adapts online.
#
# Segment construction is done in data/drift_simulator.py using strict
# index-based slicing with assert checks to prevent any class leakage.
N_ACTIONS = 5   # classes: BENIGN, DoS, DDoS, BruteForce, Other

# ── Neural network architecture ────────────────────────────────────────────
# Applied to both Deep SARSA and DQN (same QNetwork class).
# Mohamed & Ejbali (2023) don't publish exact sizes; [128, 64] is standard.
HIDDEN_DIMS = [128, 64]

# ── Shared RL hyperparameters (SARSA and DQN) ──────────────────────────────
LEARNING_RATE = 0.001
GAMMA = 0.99    # discount factor
EPSILON = 1.0     # initial exploration rate (start fully exploratory)
EPSILON_DECAY = 0.995   # multiply epsilon by this each episode
EPSILON_MIN = 0.01    # floor on exploration

REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 64
N_EPISODES = 30      # number of training episodes

# Steps sampled per episode — set to None to use the full segment each episode.
# Use a smaller number (e.g. 5000) for faster iteration on Colab/CPU.
MAX_STEPS_PER_EPISODE = 5000

# ── DQN-specific ───────────────────────────────────────────────────────────
TARGET_UPDATE_FREQ = 10      # sync target network every N steps (not episodes)
