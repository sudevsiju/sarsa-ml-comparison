# Compute and display performance metrics
# Covers all metrics specified in the project proposal evaluation plan: accuracy, precision, recall, F1-score, detection latency, and stability.

import time
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)

# evaluate accuracy, precision, recall, F1.
def evaluate(y_true, y_pred, model_name="Model"):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    print(f"  {model_name:<18} Acc={metrics['accuracy']:.4f}  "
          f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
          f"F1={metrics['f1']:.4f}")
    return metrics

def measure_latency(predict_fn, X, n_runs=100):
    times = []
    sample = X[:1]  # single sample
    for _ in range(n_runs):
        start = time.perf_counter()
        predict_fn(sample)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # convert to ms
    return float(np.mean(times))

def print_report(y_true, y_pred, label_names=None):
    print(classification_report(y_true, y_pred,
                                target_names=label_names, zero_division=0))

def compute_stability(scores):
    return float(np.var(scores))

# formatted comparison table of all models
def compare_results(results_dict):
    segments = list(results_dict.keys())
    models = list(next(iter(results_dict.values())).keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']

    col_w = 10
    header = f"{'Model':<18}"
    for seg in segments:
        for m in metrics:
            header += f"{seg[:4]+'/'+m[:3]:>{col_w}}"
    print("\n" + header)
    print("-" * len(header))

    for model in models:
        row = f"{model:<18}"
        for seg in segments:
            for m in metrics:
                val = results_dict[seg].get(model, {}).get(m, 0.0)
                row += f"{val:>{col_w}.4f}"
        print(row)

    print("\nF1 Stability (variance pre→post drift — lower is better):")
    for model in models:
        f1_scores = [results_dict[s][model]['f1'] for s in segments if model in results_dict[s]]
        var = compute_stability(f1_scores)
        drop = f1_scores[0] - f1_scores[-1] if len(f1_scores) >= 2 else 0
        print(f"  {model:<18} variance={var:.6f}  F1 drop={drop:+.4f}")
