# data/dataset_loader.py
# Loads raw CICIDS2017 CSV files, cleans them, and converts them into
# scaled NumPy arrays ready for training and evaluation.
# Preprocessed arrays are cached to disk so subsequent runs skip reprocessing.

import os
import glob
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")

# Maps CICIDS2017 label strings to integer class indices.
# Related attack variants are grouped into one class to keep the problem tractable
# while preserving the semantic differences needed for concept drift evaluation.
# 0=BENIGN, 1=DoS, 2=DDoS (excluded from T1 training), 3=BruteForce, 4=Other.
# Any label not listed here is assigned class 4 (Other) at preprocessing time.
MULTICLASS_MAP = {
    'BENIGN': 0,
    'DoS Hulk': 1,
    'DoS GoldenEye': 1,
    'DoS slowloris': 1,
    'DoS Slowhttptest': 1,
    'DDoS': 2,
    'FTP-Patator': 3,
    'SSH-Patator': 3,
}
CLASS_NAMES = ['BENIGN', 'DoS', 'DDoS', 'BruteForce', 'Other']


def save_cache(cache_dir, X_train, X_test, y_train, y_test,
               X_t2, y_t2, X_t3, y_t3, scaler, label_classes):
    """Save all preprocessed arrays to disk so future runs can skip reprocessing."""
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, "X_train.npy"), X_train)
    np.save(os.path.join(cache_dir, "X_test.npy"),  X_test)
    np.save(os.path.join(cache_dir, "y_train.npy"), y_train)
    np.save(os.path.join(cache_dir, "y_test.npy"),  y_test)
    np.save(os.path.join(cache_dir, "X_t2.npy"),    X_t2)
    np.save(os.path.join(cache_dir, "y_t2.npy"),    y_t2)
    np.save(os.path.join(cache_dir, "X_t3.npy"),    X_t3)
    np.save(os.path.join(cache_dir, "y_t3.npy"),    y_t3)
    np.save(os.path.join(cache_dir, "classes.npy"), np.array(label_classes))
    joblib.dump(scaler, os.path.join(cache_dir, "scaler.joblib"))
    print(f"  Cache saved to: {cache_dir}")


def load_cache(cache_dir=None):
    """
    Load preprocessed arrays from disk if a complete cache exists.
    Returns a tuple (X_train, X_test, y_train, y_test, X_t2, y_t2, X_t3, y_t3,
    scaler, label_classes), or None if the cache is missing or incomplete.
    """
    cache_dir = cache_dir or CACHE_DIR
    required  = ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy",
                 "X_t2.npy", "y_t2.npy", "X_t3.npy", "y_t3.npy",
                 "classes.npy", "scaler.joblib"]

    if not all(os.path.exists(os.path.join(cache_dir, f)) for f in required):
        return None

    print(f"  Loading preprocessed data from cache: {cache_dir}")
    X_train = np.load(os.path.join(cache_dir, "X_train.npy"))
    X_test = np.load(os.path.join(cache_dir, "X_test.npy"))
    y_train = np.load(os.path.join(cache_dir, "y_train.npy"))
    y_test = np.load(os.path.join(cache_dir, "y_test.npy"))
    X_t2 = np.load(os.path.join(cache_dir, "X_t2.npy"))
    y_t2 = np.load(os.path.join(cache_dir, "y_t2.npy"))
    X_t3 = np.load(os.path.join(cache_dir, "X_t3.npy"))
    y_t3 = np.load(os.path.join(cache_dir, "y_t3.npy"))
    label_classes = list(np.load(os.path.join(cache_dir, "classes.npy")))
    scaler = joblib.load(os.path.join(cache_dir, "scaler.joblib"))
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}  "
          f"T2: {X_t2.shape}  T3: {X_t3.shape}")
    return X_train, X_test, y_train, y_test, X_t2, y_t2, X_t3, y_t3, scaler, label_classes


def load_cicids2017(data_dir):
    """
    Read all CSV files in data_dir and concatenate into a single DataFrame.
    Handles two known CICIDS2017 issues: whitespace in column names and
    mixed UTF-8/Latin-1 file encodings.
    """
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not all_files:
        raise FileNotFoundError(
            f"No CSV files found in: {data_dir}\n"
            f"Download CICIDS2017 and place the CSV files in that directory."
        )

    dfs = []
    for f in sorted(all_files):
        try:
            df = pd.read_csv(f, encoding='utf-8', low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(f, encoding='latin-1', low_memory=False)
        df.columns = df.columns.str.strip()
        print(f"  Loaded {os.path.basename(f)}: {len(df):,} rows")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Combined: {len(combined):,} records, {combined.shape[1]} columns")
    return combined


def preprocess(df, scaler=None):
    """
    Clean the raw DataFrame and return scaled features with multiclass labels.

    Cleaning removes infinite values (a known CICIDS2017 artefact), NaN rows,
    and duplicate rows. Labels are mapped via MULTICLASS_MAP; unknown labels
    become class 4 (Other). Using multiclass labels rather than binary BENIGN/ATTACK
    is what makes concept drift meaningful — DDoS in T3 carries label 2, which
    no model has ever been trained to predict, forcing static models to fail.

    The scaler is fitted once on the full dataset and reused across all segments
    so every input shares the same numerical scale, as it would in deployment.

    Args:
        df:     raw DataFrame from load_cicids2017()
        scaler: pre-fitted StandardScaler to reuse, or None to fit a new one

    Returns:
        X, y, label_classes, scaler
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    label_col = 'Label'
    df[label_col] = df[label_col].str.strip()

    # CICIDS2017 contains literal "Infinity" strings as well as numpy inf values
    df.replace([np.inf, -np.inf, 'Infinity', '-Infinity'], np.nan, inplace=True)
    before = len(df)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    print(f"  Cleaned: {before:,} → {len(df):,} rows")

    # Labels absent from MULTICLASS_MAP become class 4 (Other)
    y = df[label_col].map(MULTICLASS_MAP).fillna(4).astype(np.int64).values

    feature_cols = [c for c in df.columns if c != label_col]
    X = df[feature_cols].select_dtypes(include=[np.number])

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled.astype(np.float32), y, CLASS_NAMES, scaler


def get_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Stratified train/test split that preserves the class ratio in both halves.
    Stratification prevents rare attack classes from being lost from the test set.
    """
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)
