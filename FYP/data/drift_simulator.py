# data/drift_simulator.py
# Splits the preprocessed CICIDS2017 dataset into three evaluation segments
# that simulate increasing levels of concept drift.
# Strict class separation is enforced with assert statements — if a class leaks
# into the wrong segment the script stops immediately and reports where.

import numpy as np

# Integer class labels — must stay in sync with MULTICLASS_MAP in dataset_loader.py
BENIGN     = 0  # normal traffic
DOS        = 1  # DoS attacks (Hulk, GoldenEye, slowloris, Slowhttptest)
DDOS       = 2  # Distributed DoS — the unseen class introduced at T3
BRUTEFORCE = 3  # FTP-Patator and SSH-Patator brute-force attacks
OTHER      = 4  # all remaining attack types


def create_strict_segments(X, y, random_state=42):
    """
    Partition the dataset into three segments of increasing concept drift severity.

    T1 (no drift): BENIGN + DoS only. All models are trained on this distribution.
    DDoS and BruteForce are excluded so models have no prior knowledge of them.

    T2 (mild drift): BENIGN + DoS (different samples) + BruteForce.
    BruteForce is new — a moderate shift from the training distribution.

    T3 (severe drift): BENIGN + DDoS only. DDoS is completely absent from T1,
    so static models must classify an attack type they have never encountered.
    The segment is balanced at ~60% BENIGN / 40% DDoS to prevent models from
    scoring well simply by predicting everything as BENIGN.

    Args:
        X:            np.ndarray (n_samples, n_features) — scaled features
        y:            np.ndarray (n_samples,) — integer class labels (0-4)
        random_state: seed for reproducible shuffling

    Returns:
        (X_t1, y_t1), (X_t2, y_t2), (X_t3, y_t3)
    """
    rng = np.random.default_rng(random_state)

    idx_benign = np.where(y == BENIGN)[0]
    idx_dos = np.where(y == DOS)[0]
    idx_ddos = np.where(y == DDOS)[0]
    idx_brute = np.where(y == BRUTEFORCE)[0]

    print(f"\n  Class counts in full dataset:")
    print(f"    BENIGN:     {len(idx_benign):>7,}")
    print(f"    DoS:        {len(idx_dos):>7,}")
    print(f"    DDoS:       {len(idx_ddos):>7,}")
    print(f"    BruteForce: {len(idx_brute):>7,}")
    print(f"    Other:      {np.sum(y == OTHER):>7,}")

    # Cap class sizes to keep training time manageable on CPU
    n_dos = min(len(idx_dos), 20000)
    n_brute = min(len(idx_brute), 5000)

    # Shuffle each class independently so slices are random samples,
    # not ordered by time-of-capture (which would introduce temporal bias)
    rng.shuffle(idx_benign)
    rng.shuffle(idx_dos)
    rng.shuffle(idx_ddos)
    rng.shuffle(idx_brute)

    # T1: 20,000 BENIGN + up to 20,000 DoS
    # Each segment draws from a different slice of the shuffled index arrays
    # so there is no sample overlap between segments
    t1_idx = np.concatenate([idx_benign[:20000], idx_dos[:n_dos]])
    rng.shuffle(t1_idx)

    assert np.sum(y[t1_idx] == DDOS)       == 0, "LEAK: DDoS found in T1!"
    assert np.sum(y[t1_idx] == BRUTEFORCE) == 0, "LEAK: BruteForce found in T1!"
    print(f"\n  T1 verified — classes present: {np.unique(y[t1_idx]).tolist()}  "
          f"size={len(t1_idx):,}")

    # T2: 20,000 BENIGN (indices 20k-40k) + up to 6,000 DoS + up to 5,000 BruteForce
    t2_idx = np.concatenate([
        idx_benign[20000:40000],
        idx_dos[n_dos: n_dos + min(6000, len(idx_dos) - n_dos)],
        idx_brute[:n_brute],
    ])
    rng.shuffle(t2_idx)

    assert np.sum(y[t2_idx] == DDOS) == 0, "LEAK: DDoS found in T2!"
    print(f"  T2 verified — classes present: {np.unique(y[t2_idx]).tolist()}  "
          f"size={len(t2_idx):,}")

    # T3: ~12,000 BENIGN + 8,000 DDoS (~60/40 split)
    # A heavily skewed dataset would let a naive model score ~0.95 by always
    # predicting BENIGN, masking the real performance collapse under drift
    n_t3_ddos = min(len(idx_ddos), 8000)
    n_t3_benign = int(n_t3_ddos * 1.5)
    t3_idx = np.concatenate([
        idx_benign[40000: 40000 + n_t3_benign],
        idx_ddos[:n_t3_ddos],
    ])
    rng.shuffle(t3_idx)

    assert np.sum(y[t3_idx] == DOS)        == 0, "LEAK: DoS found in T3!"
    assert np.sum(y[t3_idx] == BRUTEFORCE) == 0, "LEAK: BruteForce found in T3!"
    print(f"  T3 verified — classes present: {np.unique(y[t3_idx]).tolist()}  "
          f"size={len(t3_idx):,}")

    return (X[t1_idx], y[t1_idx]), (X[t2_idx], y[t2_idx]), (X[t3_idx], y[t3_idx])


def get_segment_summary(y_t1, y_t2, y_t3):
    """Print the class distribution for each segment as a sanity check."""
    from data.dataset_loader import CLASS_NAMES
    for name, y in [("T1 — No Drift",    y_t1),
                    ("T2 — Mild Drift",   y_t2),
                    ("T3 — Severe Drift", y_t3)]:
        print(f"\n=== {name} ===")
        for cls_id, cls_name in enumerate(CLASS_NAMES):
            count = np.sum(y == cls_id)
            if count > 0:
                print(f"  {cls_name:<12}: {count:>7,}  ({100 * count / len(y):.1f}%)")
        print(f"  Total: {len(y):,}")
