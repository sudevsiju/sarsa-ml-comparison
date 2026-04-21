# models/baselines.py — traditional ML baseline (Random Forest)
#
# Trained once on the pre-drift segment and evaluated on both segments.
# Expected result: high performance pre-drift, noticeable drop post-drift,
# because static ML models cannot adapt to unseen attack patterns.
# Reference: Lu et al. (2024). "Research on intrusion detection based on an
# enhanced random forest algorithm." Applied Sciences, 14(2), p.714.

from sklearn.ensemble import RandomForestClassifier


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest classifier.

    Args:
        X_train:      np.ndarray of scaled features
        y_train:      np.ndarray of encoded labels
        n_estimators: number of trees (100 is standard)
        random_state: seed for reproducibility

    Returns:
        fitted RandomForestClassifier (has .predict() and .predict_proba())
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1  # use all CPU cores
    )
    clf.fit(X_train, y_train)
    return clf
