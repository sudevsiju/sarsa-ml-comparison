# Splits the preprocessed CICIDS2017 dataset into three evaluation segments that simulate increasing levels of concept drift.

import numpy as np

# Integer class labels — must stay in sync with MULTICLASS_MAP in dataset_loader.py
BENIGN = 0  # normal traffic
DOS = 1  # DoS attacks (Hulk, GoldenEye, slowloris, Slowhttptest)
DDOS = 2  # Distributed DoS — the unseen class introduced at T3
BRUTEFORCE = 3  # FTP-Patator and SSH-Patator brute-force attacks
OTHER = 4  # all remaining attack types

# Partition the dataset into three segments
def create_strict_segments(X, y, random_state=42):
    rng = np.random.default_rng(random_state)

    idx_benign = np.where(y == BENIGN)[0]
    idx_dos = np.where(y == DOS)[0]
    idx_ddos = np.where(y == DDOS)[0]
    idx_brute = np.where(y == BRUTEFORCE)[0]

    print(f"\n  Class counts in full dataset:")
    print(f"BENIGN: {len(idx_benign):>7,}")
    print(f"DoS: {len(idx_dos):>7,}")
    print(f"DDoS: {len(idx_ddos):>7,}")
    print(f"BruteForce: {len(idx_brute):>7,}")
    print(f"Other: {np.sum(y == OTHER):>7,}")

    # Cap class sizes to keep training time manageable on CPU
    n_dos = min(len(idx_dos), 20000)
    n_brute = min(len(idx_brute), 5000)

    # Shuffle each class
    rng.shuffle(idx_benign)
    rng.shuffle(idx_dos)
    rng.shuffle(idx_ddos)
    rng.shuffle(idx_brute)

    t1_idx = np.concatenate([idx_benign[:20000], idx_dos[:n_dos]])
    rng.shuffle(t1_idx)

    assert np.sum(y[t1_idx] == DDOS) == 0, "LEAK: DDoS found in T1!"
    assert np.sum(y[t1_idx] == BRUTEFORCE) == 0, "LEAK: BruteForce found in T1!"
    print(f"\n  T1 verified — classes present: {np.unique(y[t1_idx]).tolist()}  "
          f"size={len(t1_idx):,}")

    t2_idx = np.concatenate([
        idx_benign[20000:40000],
        idx_dos[n_dos: n_dos + min(6000, len(idx_dos) - n_dos)],
        idx_brute[:n_brute],
    ])
    rng.shuffle(t2_idx)

    assert np.sum(y[t2_idx] == DDOS) == 0, "LEAK: DDoS found in T2!"
    print(f"T2 verified — classes present: {np.unique(y[t2_idx]).tolist()}  "
          f"size={len(t2_idx):,}")

    n_t3_ddos = min(len(idx_ddos), 8000)
    n_t3_benign = int(n_t3_ddos * 1.5)
    t3_idx = np.concatenate([
        idx_benign[40000: 40000 + n_t3_benign],
        idx_ddos[:n_t3_ddos],
    ])
    rng.shuffle(t3_idx)

    assert np.sum(y[t3_idx] == DOS) == 0, "LEAK: DoS found in T3!"
    assert np.sum(y[t3_idx] == BRUTEFORCE) == 0, "LEAK: BruteForce found in T3!"
    print(f"T3 verified — classes present: {np.unique(y[t3_idx]).tolist()}  "
          f"size={len(t3_idx):,}")

    return (X[t1_idx], y[t1_idx]), (X[t2_idx], y[t2_idx]), (X[t3_idx], y[t3_idx])


def get_segment_summary(y_t1, y_t2, y_t3):
    from data.dataset_loader import CLASS_NAMES
    for name, y in [("T1 — No Drift",    y_t1),
                    ("T2 — Mild Drift",   y_t2),
                    ("T3 — Severe Drift", y_t3)]:
        print(f"\n=== {name} ===")
        for cls_id, cls_name in enumerate(CLASS_NAMES):
            count = np.sum(y == cls_id)
            if count > 0:
                print(f"  {cls_name:<12}: {count:>7,}  ({100 * count / len(y):.1f}%)")
        print(f"Total: {len(y):,}")
