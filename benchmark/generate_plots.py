#!/usr/bin/env python3
"""Generate README-quality benchmark plots."""

import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

CSV_PATH = "benchmark/results/full_benchmark.csv"
OUT_DIR = "benchmark/results"

VERSION_ORDER = ["v0_cpu", "v1_naive", "v2_tiled", "v3_fused",
                 "v4_streams", "v5_mixed", "v6_tensor"]
VERSION_LABELS = {
    "v0_cpu":     "V0: CPU",
    "v1_naive":   "V1: cuBLAS",
    "v2_tiled":   "V2: Tiled",
    "v3_fused":   "V3: Fused",
    "v4_streams": "V4: Streams",
    "v5_mixed":   "V5: FP16",
    "v6_tensor":  "V6: Tensor",
}
VERSION_COLORS = {
    "v0_cpu":     "#6c757d",
    "v1_naive":   "#0d6efd",
    "v2_tiled":   "#198754",
    "v3_fused":   "#fd7e14",
    "v4_streams": "#6f42c1",
    "v5_mixed":   "#20c997",
    "v6_tensor":  "#dc3545",
}

def load_data():
    rows = []
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            r["epoch"] = int(r["epoch"])
            r["loss"] = float(r["loss"])
            r["accuracy"] = float(r["accuracy"])
            r["time_ms"] = float(r["time_ms"])
            rows.append(r)
    return rows

def get_avg_time(rows, version, config, activation):
    """Get average epoch time for final 7 epochs (skip warmup)."""
    data = [r for r in rows if r["version"]==version and r["config"]==config and r["activation"]==activation]
    if not data: return None
    # Use last 7 epochs or all if < 7
    data.sort(key=lambda x: x["epoch"])
    use = data[-min(7, len(data)):]
    return sum(d["time_ms"] for d in use) / len(use)

def get_final_accuracy(rows, version, config, activation):
    data = [r for r in rows if r["version"]==version and r["config"]==config and r["activation"]==activation]
    if not data: return None
    data.sort(key=lambda x: x["epoch"])
    return data[-1]["accuracy"]

# ===================================================================
# Plot 1: Speedup bar chart (medium network, both activations)
# ===================================================================
def plot_speedup(rows):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor('#0d1117')

    for idx, act in enumerate(["sigmoid", "relu"]):
        ax = axes[idx]
        ax.set_facecolor('#161b22')

        cpu_time = get_avg_time(rows, "v0_cpu", "medium", act)
        if not cpu_time: continue

        versions = [v for v in VERSION_ORDER if v != "v0_cpu"]
        speedups = []
        labels = []
        colors = []

        for v in versions:
            t = get_avg_time(rows, v, "medium", act)
            if t:
                speedups.append(cpu_time / t)
                labels.append(VERSION_LABELS[v])
                colors.append(VERSION_COLORS[v])

        bars = ax.bar(range(len(labels)), speedups, color=colors,
                      edgecolor='#30363d', linewidth=1.2, width=0.65)

        for bar, sp in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                    f"{sp:.0f}x", ha="center", va="bottom",
                    fontsize=11, fontweight="bold", color="#e6edf3")

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9, color="#8b949e")
        ax.set_ylabel("Speedup vs CPU", fontsize=11, color="#e6edf3")
        title_act = "Sigmoid" if act == "sigmoid" else "ReLU"
        ax.set_title(f"784→512→256→10 ({title_act})",
                     fontsize=13, fontweight="bold", color="#e6edf3")
        ax.tick_params(axis='y', colors='#8b949e')
        ax.spines['bottom'].set_color('#30363d')
        ax.spines['left'].set_color('#30363d')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.15, color='#8b949e')

    fig.suptitle("GPU Speedup Over Sequential CPU Baseline",
                 fontsize=16, fontweight="bold", color="#e6edf3", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "speedup_chart.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor='#0d1117')
    plt.close()
    print(f"  -> {path}")

# ===================================================================
# Plot 2: Training time comparison (log scale, both configs)
# ===================================================================
def plot_epoch_times(rows):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor('#0d1117')

    for idx, cfg in enumerate(["small", "medium"]):
        ax = axes[idx]
        ax.set_facecolor('#161b22')

        bar_width = 0.35
        versions = VERSION_ORDER
        sigmoid_times = []
        relu_times = []
        labels = []

        for v in versions:
            s = get_avg_time(rows, v, cfg, "sigmoid")
            r = get_avg_time(rows, v, cfg, "relu")
            if s and r:
                sigmoid_times.append(s)
                relu_times.append(r)
                labels.append(VERSION_LABELS[v])

        x = np.arange(len(labels))
        ax.bar(x - bar_width/2, sigmoid_times, bar_width, label="Sigmoid",
               color="#6366f1", edgecolor='#30363d')
        ax.bar(x + bar_width/2, relu_times, bar_width, label="ReLU",
               color="#f97316", edgecolor='#30363d')

        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=25, ha="right", color="#8b949e")
        ax.set_ylabel("Epoch Time (ms, log scale)", fontsize=10, color="#e6edf3")

        cfg_label = "784→128→10" if cfg == "small" else "784→512→256→10"
        ax.set_title(f"{cfg_label}", fontsize=13, fontweight="bold", color="#e6edf3")
        ax.legend(fontsize=9, facecolor='#161b22', edgecolor='#30363d',
                  labelcolor='#e6edf3')
        ax.tick_params(axis='y', colors='#8b949e')
        ax.spines['bottom'].set_color('#30363d')
        ax.spines['left'].set_color('#30363d')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.15, color='#8b949e')

    fig.suptitle("Training Time per Epoch (Lower is Better)",
                 fontsize=16, fontweight="bold", color="#e6edf3", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "epoch_times.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor='#0d1117')
    plt.close()
    print(f"  -> {path}")

# ===================================================================
# Plot 3: Convergence curves (accuracy over epochs)
# ===================================================================
def plot_convergence(rows):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor('#0d1117')

    configs = [("small", "relu", "784→128→10 (ReLU)"),
               ("medium", "relu", "784→512→256→10 (ReLU)")]

    for idx, (cfg, act, title) in enumerate(configs):
        ax = axes[idx]
        ax.set_facecolor('#161b22')

        for v in VERSION_ORDER:
            data = [r for r in rows if r["version"]==v and r["config"]==cfg and r["activation"]==act]
            if not data: continue
            data.sort(key=lambda x: x["epoch"])
            epochs = [d["epoch"] for d in data]
            accs = [d["accuracy"] for d in data]
            ax.plot(epochs, accs, '-o', label=VERSION_LABELS[v],
                    color=VERSION_COLORS[v], markersize=3, linewidth=1.8)

        ax.set_xlabel("Epoch", fontsize=10, color="#e6edf3")
        ax.set_ylabel("Test Accuracy (%)", fontsize=10, color="#e6edf3")
        ax.set_title(title, fontsize=13, fontweight="bold", color="#e6edf3")
        ax.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d',
                  labelcolor='#e6edf3', ncol=2)
        ax.tick_params(axis='both', colors='#8b949e')
        ax.spines['bottom'].set_color('#30363d')
        ax.spines['left'].set_color('#30363d')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.15, color='#8b949e')

    fig.suptitle("Convergence: All Versions Reach Same Accuracy",
                 fontsize=16, fontweight="bold", color="#e6edf3", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "convergence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor='#0d1117')
    plt.close()
    print(f"  -> {path}")

# ===================================================================
# Plot 4: GPU-only comparison (exclude CPU, linear scale)
# ===================================================================
def plot_gpu_comparison(rows):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    gpu_versions = [v for v in VERSION_ORDER if v != "v0_cpu"]
    bar_width = 0.2
    configs = [("small", "sigmoid"), ("small", "relu"),
               ("medium", "sigmoid"), ("medium", "relu")]
    x = np.arange(len(gpu_versions))

    config_labels = ["Small/Sigmoid", "Small/ReLU", "Medium/Sigmoid", "Medium/ReLU"]
    config_colors = ["#6366f1", "#f97316", "#22c55e", "#ef4444"]

    for i, (cfg, act) in enumerate(configs):
        times = []
        for v in gpu_versions:
            t = get_avg_time(rows, v, cfg, act)
            times.append(t if t else 0)
        ax.bar(x + i*bar_width - 1.5*bar_width, times, bar_width,
               label=config_labels[i], color=config_colors[i],
               edgecolor='#30363d', linewidth=0.8)

    labels = [VERSION_LABELS[v] for v in gpu_versions]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, color="#8b949e")
    ax.set_ylabel("Epoch Time (ms)", fontsize=11, color="#e6edf3")
    ax.set_title("GPU Versions Detailed Comparison",
                 fontsize=14, fontweight="bold", color="#e6edf3")
    ax.legend(fontsize=9, facecolor='#161b22', edgecolor='#30363d',
              labelcolor='#e6edf3', ncol=2)
    ax.tick_params(axis='y', colors='#8b949e')
    ax.spines['bottom'].set_color('#30363d')
    ax.spines['left'].set_color('#30363d')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.15, color='#8b949e')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "gpu_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor='#0d1117')
    plt.close()
    print(f"  -> {path}")

def main():
    rows = load_data()
    print(f"Loaded {len(rows)} data points\n")
    plot_speedup(rows)
    plot_epoch_times(rows)
    plot_convergence(rows)
    plot_gpu_comparison(rows)
    print("\nAll plots generated!")

if __name__ == "__main__":
    main()
