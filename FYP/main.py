# top-level entry point
# Run this file to execute the full pipeline:
#   1. Load and preprocess CICIDS2017
#   2. Create concept drift segments
#   3. Train all models on segment 1 (pre-drift)
#   4. Evaluate all models on segment 1 (pre-drift) and segment 2 (post-drift)
#   5. Measure detection latency
#   6. Print comparison table and save plots

import os
import time
import numpy as np

import config
from data.dataset_loader import (load_cicids2017, preprocess, get_train_test_split,
                                    save_cache, load_cache)
from data.drift_simulator import create_strict_segments, get_segment_summary
from environment.ids_env import IDSEnvironment
from models.deep_sarsa import DeepSARSA
from models.dqn import DQN
from models.baselines import train_random_forest
from evaluation.metrics import evaluate, measure_latency, compare_results, print_report
from evaluation.visualizer import (plot_f1_over_segments, plot_metric_comparison,
                                    plot_confusion_matrix, plot_latency,
                                    plot_training_rewards)


# Training loops 
def train_sarsa(agent, X_train, y_train, n_episodes, max_steps=None):
    
    reward_history = []
    ep_times = []

    for ep in range(1, n_episodes + 1):
        ep_start = time.time()
        idx  = np.random.permutation(len(X_train))
        X_ep = X_train[idx[:max_steps]] if max_steps else X_train[idx]
        y_ep = y_train[idx[:max_steps]] if max_steps else y_train[idx]

        env    = IDSEnvironment(X_ep, y_ep)
        state  = env.reset()
        action = agent.select_action(state)
        ep_rewards = []
        done = False

        while not done:
            next_state, reward, done = env.step(action)
            next_state_arr = next_state if next_state is not None else np.zeros_like(state)
            next_action = agent.select_action(next_state_arr) if not done else 0

            agent.store_transition(state, action, reward, next_state_arr, next_action)
            agent.update()
            ep_rewards.append(reward)

            if not done:
                state = next_state
                action = next_action

        agent.decay_epsilon()
        avg_r = float(np.mean(ep_rewards))
        reward_history.append(avg_r)

        ep_times.append(time.time() - ep_start)
        avg_ep_time = sum(ep_times) / len(ep_times)
        eta = avg_ep_time * (n_episodes - ep)
        print(f"[SARSA] Ep {ep:>3}/{n_episodes}  avg_reward={avg_r:.4f}  ε={agent.epsilon:.4f}"
              f"ep_time={ep_times[-1]:.1f}s  ETA={eta:.0f}s")

    return reward_history


def evaluate_sarsa_online(agent, X_eval, y_eval):

    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0        # greedy predictions; adaptation comes from weight updates

    n = len(X_eval)
    preds = []
    t_start = time.time()
    interval = max(1, n // 10)  # print progress every 10%

    for i in range(n):
        state = X_eval[i]
        action = agent.select_action(state)
        preds.append(action)

        reward = 1.0 if action == int(y_eval[i]) else -1.0

        if i + 1 < n:
            next_state = X_eval[i + 1]
            next_action = agent.select_action(next_state)
        else:
            next_state  = np.zeros_like(state)
            next_action = 0

        agent.store_transition(state, action, reward, next_state, next_action)
        agent.update()

        if (i + 1) % interval == 0 or i + 1 == n:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(f"[SARSA] {i+1:>6}/{n}  ({100*(i+1)/n:.0f}%)  ETA={eta:.0f}s")

    agent.epsilon = saved_epsilon
    return np.array(preds)


def train_dqn(agent, X_train, y_train, n_episodes, max_steps=None):
    reward_history = []
    ep_times = []

    for ep in range(1, n_episodes + 1):
        ep_start = time.time()
        idx = np.random.permutation(len(X_train))
        X_ep = X_train[idx[:max_steps]] if max_steps else X_train[idx]
        y_ep = y_train[idx[:max_steps]] if max_steps else y_train[idx]

        env = IDSEnvironment(X_ep, y_ep)
        state = env.reset()
        ep_rewards = []
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)

            next_state_arr = next_state if next_state is not None else np.zeros_like(state)
            agent.store_transition(state, action, reward, next_state_arr, done)
            agent.update()
            ep_rewards.append(reward)

            if not done:
                state = next_state

        agent.decay_epsilon()
        avg_r = float(np.mean(ep_rewards))
        reward_history.append(avg_r)

        ep_times.append(time.time() - ep_start)
        avg_ep_time = sum(ep_times) / len(ep_times)
        eta = avg_ep_time * (n_episodes - ep)
        print(f"  [DQN]   Ep {ep:>3}/{n_episodes}  avg_reward={avg_r:.4f}  ε={agent.epsilon:.4f}"
              f"  ep_time={ep_times[-1]:.1f}s  ETA={eta:.0f}s")

    return reward_history


# Main pipeline 
def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    cached = load_cache()
    if cached:
        print("\n[1-3] Restored preprocessed data from cache (skipping load+preprocess).")
        X_train, X_test, y_train, y_test, X_t2, y_t2, X_t3, y_t3, scaler, classes = cached
    else:
        print("\n[1] Loading CICIDS2017...")
        df = load_cicids2017(config.DATA_DIR)

        print("\n[2] Preprocessing full dataset (multiclass labels)...")
        X_all, y_all, classes, scaler = preprocess(df)

        print("\n[3] Creating strict 3-stage concept drift segments...")
        (X1, y1), (X_t2, y_t2), (X_t3, y_t3) = create_strict_segments(
            X_all, y_all, random_state=config.RANDOM_STATE
        )
        get_segment_summary(y1, y_t2, y_t3)

        X_train, X_test, y_train, y_test = get_train_test_split(
            X1, y1, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
        )
        save_cache(config.CACHE_DIR, X_train, X_test, y_train, y_test,
                   X_t2, y_t2, X_t3, y_t3, scaler, classes)

    print(f"T1 train: {X_train.shape}  T1 test: {X_test.shape}")
    print(f"T2 (mild drift): {X_t2.shape}  T3 (severe drift): {X_t3.shape}")
    print(f"Classes: {classes}")

    n_features = X_train.shape[1]
    n_actions = config.N_ACTIONS 

    print("\n[4] Training Random Forest...")
    t0 = time.time()
    rf = train_random_forest(X_train, y_train)
    print(f"Done. ({time.time() - t0:.1f}s)")

    print("\n[5] Training Deep SARSA (Mohamed & Ejbali, 2023 approach)...")
    sarsa = DeepSARSA(
        input_dim=n_features,
        n_actions=n_actions,
        lr=config.LEARNING_RATE,
        gamma=config.GAMMA,
        epsilon=config.EPSILON,
        hidden_dims=config.HIDDEN_DIMS,
        buffer_size=config.REPLAY_BUFFER_SIZE,
        batch_size=config.BATCH_SIZE,
        epsilon_decay=config.EPSILON_DECAY,
        epsilon_min=config.EPSILON_MIN,
    )
    t0 = time.time()
    sarsa_rewards = train_sarsa(sarsa, X_train, y_train,
                                config.N_EPISODES, config.MAX_STEPS_PER_EPISODE)
    print(f"  Training complete. ({time.time() - t0:.1f}s)")
    sarsa.save(os.path.join(config.RESULTS_DIR, "sarsa_model.pt"))

    print("\n[6] Training DQN (off-policy RL baseline)...")
    dqn = DQN(
        input_dim=n_features,
        n_actions=n_actions,
        lr=config.LEARNING_RATE,
        gamma=config.GAMMA,
        epsilon=config.EPSILON,
        hidden_dims=config.HIDDEN_DIMS,
        buffer_size=config.REPLAY_BUFFER_SIZE,
        batch_size=config.BATCH_SIZE,
        target_update_freq=config.TARGET_UPDATE_FREQ,
        epsilon_decay=config.EPSILON_DECAY,
        epsilon_min=config.EPSILON_MIN,
    )
    t0 = time.time()
    dqn_rewards = train_dqn(dqn, X_train, y_train,
                            config.N_EPISODES, config.MAX_STEPS_PER_EPISODE)
    print(f"  Training complete. ({time.time() - t0:.1f}s)")
    dqn.save(os.path.join(config.RESULTS_DIR, "dqn_model.pt"))

    print("\n[7] Evaluating all models...")
    all_results = {}
    all_preds = {}

    for seg_name, X_eval, y_eval in [("T1 (no drift)", X_test, y_test),("T2 (mild drift)", X_t2,   y_t2),("T3 (severe drift)", X_t3,   y_t3)]:
        print(f"\n  --- {seg_name} ---")
        t0 = time.time()
        rf_pred = rf.predict(X_eval)
        print(f"RF eval: {time.time() - t0:.1f}s")

        t0 = time.time()
        dqn_pred  = dqn.predict(X_eval)
        print(f"DQN eval: {time.time() - t0:.1f}s")

        t0 = time.time()
        if seg_name == "T1 (no drift)":
            sarsa_pred = sarsa.predict(X_eval)
        else:
            print("[SARSA] Online adaptation enabled for drift segment...")
            sarsa_pred = evaluate_sarsa_online(sarsa, X_eval, y_eval)
        print(f"SARSA eval: {time.time() - t0:.1f}s")

        all_results[seg_name] = {
            "Random Forest": evaluate(y_eval, rf_pred, "Random Forest"),
            "DQN": evaluate(y_eval, dqn_pred,   "DQN"),
            "Deep SARSA": evaluate(y_eval, sarsa_pred, "Deep SARSA"),
        }
        all_preds[seg_name] = {
            "Random Forest": rf_pred,
            "DQN": dqn_pred,
            "Deep SARSA": sarsa_pred,
        }
 
    print("\n[8] Measuring detection latency...")
    latency_dict = {
        "Random Forest": measure_latency(rf.predict, X_test),
        "DQN": measure_latency(dqn.predict, X_test),
        "Deep SARSA": measure_latency(sarsa.predict, X_test),
    }
    for name, lat in latency_dict.items():
        print(f"  {name:<18} {lat:.4f} ms/sample")

    print("\n[9] Full results comparison:")
    compare_results(all_results)

    # Plots 
    print("\n[10] Saving plots to", config.RESULTS_DIR)

    model_seg_results = {
        m: {s: all_results[s][m] for s in all_results}
        for m in ["Random Forest", "DQN", "Deep SARSA"]
    }

    plot_f1_over_segments(
        model_seg_results,
        save_path=os.path.join(config.RESULTS_DIR, "f1_over_segments.png")
    )
    plot_metric_comparison(
        all_results, metric='f1',
        save_path=os.path.join(config.RESULTS_DIR, "f1_comparison.png")
    )
    plot_latency(
        latency_dict,
        save_path=os.path.join(config.RESULTS_DIR, "latency.png")
    )
    plot_training_rewards(
        sarsa_rewards, "Deep SARSA",
        save_path=os.path.join(config.RESULTS_DIR, "sarsa_training.png")
    )
    plot_training_rewards(
        dqn_rewards, "DQN",
        save_path=os.path.join(config.RESULTS_DIR, "dqn_training.png")
    )

    # Confusion matrices 
    for seg_name, y_eval in [("T1 (no drift)", y_test), ("T2 (mild drift)", y_t2), ("T3 (severe drift)", y_t3)]:
        plot_confusion_matrix(
            y_eval, all_preds[seg_name]["Deep SARSA"],
            class_names=classes,
            title=f"Deep SARSA — {seg_name}",
            save_path=os.path.join(config.RESULTS_DIR,
                                   f"cm_sarsa_{seg_name.split()[0]}.png")
        )

    print(f"\nDone. All results saved to: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
