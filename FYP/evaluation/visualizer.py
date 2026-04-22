# Generate dissertation plots
# Produces all figures needed for the project report:
# F1 score before/after concept drift (key result plot)
# Confusion matrices per model per segment
# Bar chart comparing all metrics across models
# Detection latency comparison
# Training reward curves

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Grouped bar chart
def plot_f1_over_segments(segment_results, save_path=None): 
    models = list(segment_results.keys())
    segments = list(next(iter(segment_results.values())).keys())
    x = np.arange(len(models))
    width = 0.8 / len(segments)
    colours = ['steelblue', 'tomato', 'seagreen', 'orange', 'purple'][:len(segments)]

    fig, ax = plt.subplots(figsize=(9, 5))
    offset = (len(segments) - 1) / 2
    for i, (seg, col) in enumerate(zip(segments, colours)):
        vals = [segment_results[m][seg]['f1'] for m in models]
        bars = ax.bar(x + (i - offset) * width, vals, width, label=seg, color=col, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score Before and After Concept Drift')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _save_and_show(fig, save_path)

# Heatmap 
def plot_confusion_matrix(y_true, y_pred, class_names, title="", save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title or 'Confusion Matrix')

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black', fontsize=9)

    plt.tight_layout()
    _save_and_show(fig, save_path)

# Grouped bar chart comparing one metric across all models and segments.
def plot_metric_comparison(results_by_segment, metric='f1', save_path=None):
    segments = list(results_by_segment.keys())
    models = list(next(iter(results_by_segment.values())).keys())
    x = np.arange(len(models))
    width = 0.8 / len(segments)
    colours = ['steelblue', 'tomato', 'seagreen', 'darkorange']

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (seg, col) in enumerate(zip(segments, colours)):
        offset = (i - len(segments) / 2 + 0.5) * width
        vals   = [results_by_segment[seg][m][metric] for m in models]
        ax.bar(x + offset, vals, width, label=seg, color=col, alpha=0.85)

    ax.set_ylabel(metric.capitalize())
    ax.set_title(f'{metric.upper()} Comparison — All Models')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _save_and_show(fig, save_path)

# Bar chart of average detection latency
def plot_latency(latency_dict, save_path=None):
    models = list(latency_dict.keys())
    latencies = list(latency_dict.values())

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(models, latencies, color='steelblue', alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0001,
                f'{val:.4f}ms', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Avg Latency (ms / sample)')
    ax.set_title('Detection Latency per Model')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _save_and_show(fig, save_path)

# Line chart of average reward per episode during training.
def plot_training_rewards(reward_history, model_name, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(reward_history, color='steelblue', linewidth=1.5, label=model_name)
    ax.axhline(y=0, color='grey', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg Reward')
    ax.set_title(f'Training Reward Curve — {model_name}')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_and_show(fig, save_path)

def _save_and_show(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()
    plt.close(fig)
