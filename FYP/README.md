# Deep SARSA-Based Intrusion Detection System Under Concept Drift

BSc Cybersecurity Final Year Project  
University of Greenwich  
Student: Sudev Siju

---

## Overview

This project implements and evaluates a Deep SARSA reinforcement learning agent for network intrusion detection under simulated concept drift, using the CICIDS2017 dataset. It is compared against two baselines: a Random Forest classifier and a Deep Q-Network (DQN) agent.

The core research question is whether an on-policy RL agent (Deep SARSA) can maintain higher classification performance than static or off-policy models when the traffic distribution shifts after deployment.

---

## Project Structure

```
FYP/
├── main.py                    # Entry point — runs the full pipeline
├── config.py                  # All hyperparameters and file paths
├── data/
│   ├── dataset_loader.py      # Loads and preprocesses CICIDS2017 CSV files
│   ├── drift_simulator.py     # Splits data into T1/T2/T3 drift segments
│   └── cache/                 # Auto-generated preprocessed arrays (ignored by git)
├── environment/
│   └── ids_env.py             # RL environment wrapping the dataset
├── models/
│   ├── deep_sarsa.py          # Deep SARSA agent (Mohamed & Ejbali, 2023)
│   ├── dqn.py                 # DQN baseline agent
│   └── baselines.py           # Random Forest baseline
├── evaluation/
│   ├── metrics.py             # Accuracy, precision, recall, F1, latency
│   └── visualizer.py          # All plots saved to results/
└── results/                   # Auto-generated figures and saved models
```

---

## Requirements

Python 3.10 or later is recommended.

Install dependencies:

```
pip install torch numpy pandas scikit-learn matplotlib joblib
```

Library versions used during development:

| Library | Version |
|---|---|
| torch | 2.11.0 |
| numpy | 2.3.2 |
| pandas | 3.0.2 |
| scikit-learn | 1.8.0 |
| matplotlib | 3.10.5 |
| joblib | 1.5.3 |

---

## Dataset

This project uses the CICIDS2017 dataset (Canadian Institute for Cybersecurity).

Download it from Kaggle: https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset

After downloading, update the `DATA_DIR` path in `config.py` to point to the folder containing the CSV files:

```python
DATA_DIR = r"path\to\your\cicids2017\csvs"
```

---

## How to Run

```
python main.py
```

The pipeline runs in order:

1. Loads and preprocesses CICIDS2017 (skipped on subsequent runs if cache exists)
2. Splits data into three concept drift segments (T1, T2, T3)
3. Trains Random Forest, Deep SARSA, and DQN on T1
4. Evaluates all models on T1, T2, and T3
5. Measures detection latency
6. Saves results and plots to the `results/` folder

On first run, preprocessing takes several minutes. Subsequent runs load from cache and go straight to training.

---

## Concept Drift Segments

| Segment | Contents | Purpose |
|---|---|---|
| T1 — No drift | BENIGN + DoS | Training distribution |
| T2 — Mild drift | BENIGN + DoS + BruteForce | BruteForce is a new, unseen class |
| T3 — Severe drift | BENIGN + DDoS only | DDoS is completely unseen by all models |

T3 is balanced at approximately 60% BENIGN / 40% DDoS. This is intentional — a heavily skewed dataset would allow a naive model to score well simply by predicting BENIGN for everything, which would hide the performance collapse under drift.

---

## Key Design Decisions

**Multiclass labels** — Labels are grouped into five classes (BENIGN, DoS, DDoS, BruteForce, Other). Using multiclass rather than binary labels is what makes concept drift meaningful: DDoS in T3 carries a class label (2) that no model has ever been trained to predict.

**Online adaptation** — During T2 and T3 evaluation, Deep SARSA continues updating its weights using the true label as a reward signal after each prediction. Random Forest and DQN are frozen at test time. This simulates a deployed IDS receiving ground-truth feedback from a SIEM or analyst.

**Reward shaping** — The environment penalises missed attacks and false alarms equally (-1.0), with a slightly lower penalty for predicting the wrong attack class (-0.8), since flagging malicious traffic of any kind is better than missing it entirely.

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Hidden layers | [128, 64] |
| Learning rate | 0.001 |
| Discount factor (gamma) | 0.99 |
| Initial epsilon | 1.0 |
| Epsilon decay | 0.995 |
| Epsilon minimum | 0.01 |
| Replay buffer size | 10,000 |
| Batch size | 64 |
| Training episodes | 30 |
| Steps per episode | 5,000 |
| DQN target update frequency | Every 10 steps |

---

## References

Mohamed, S. and Ejbali, R. (2023). Deep SARSA-based reinforcement learning approach for anomaly network intrusion detection system. *International Journal of Information Security*, 22(1), pp.235-247.

Sharafaldin, I., Lashkari, A.H. and Ghorbani, A.A. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. *ICISSP*, pp.108-116.
