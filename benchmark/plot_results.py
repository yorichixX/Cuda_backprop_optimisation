#!/usr/bin/env python3
"""
Generate benchmark comparison plots from CSV results.

Usage:
    python plot_results.py results/benchmark_YYYYMMDD_HHMMSS.csv

Generates:
    1. Speedup bar chart (all versions vs CPU baseline)
    2. Training time per epoch comparison
    3. Convergence curves (accuracy over epochs)
    4. Performance scaling (time vs network size)
"""

import sys
import os
import csv
import argparse
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")


VERSION_LABELS = {
    "v0_cpu":     "V0: CPU",
    "v1_naive":   "V1: cuBLAS",
    "v2_tiled":   "V2: Tiled GEMM",
    "v3_fused":   "V3: Fused",
    "v4_streams": "V4: Streams",
    "v5_mixed":   "V5: FP16",
    "v6_tensor":  "V6: Tensor Core",
}

VERSION_COLORS = {
    "v0_cpu":     "#e74c3c",
    "v1_naive":   "#3498db",
    "v2_tiled":   "#2ecc71",
    "v3_fused":   "#f39c12",
    "v4_streams": "#9b59b6",
    "v5_mixed":   "#1abc9c",
    "v6_tensor":  "#e67e22",
}

VERSION_ORDER = ["v0_cpu", "v1_naive", "v2_tiled", "v3_fused",
                 "v4_streams", "v5_mixed", "v6_tensor"]


def load_csv(path):
    """Load benchmark results from CSV."""
    results = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["epoch"] = int(row["epoch"])
            row["loss"] = float(row["loss"])
            row["accuracy"] = float(row["accuracy"])
            row["time_ms"] = float(row["time_ms"])
            results.append(row)
    return results


def plot_speedup(results, output_dir):
    """Bar chart: speedup of each version over CPU baseline."""
    # Group by config and version, compute average epoch time
    avg_times = defaultdict(dict)
    for r in results:
        key = (r["version"], r["config"])
        if key not in avg_times:
            avg_times[key] = []
        avg_times[key].append(r["time_ms"])

    configs = sorted(set(r["config"] for r in results))

    fig, ax = plt.subplots(figsize=(12, 6))

    for config in configs:
        # Get CPU baseline time
        cpu_key = ("v0_cpu", config)
        if cpu_key not in avg_times:
            continue
        cpu_time = sum(avg_times[cpu_key]) / len(avg_times[cpu_key])

        versions = [v for v in VERSION_ORDER if (v, config) in avg_times and v != "v0_cpu"]
        speedups = []
        labels = []
        colors = []

        for v in versions:
            vk = (v, config)
            vtime = sum(avg_times[vk]) / len(avg_times[vk])
            speedups.append(cpu_time / vtime)
            labels.append(VERSION_LABELS.get(v, v))
            colors.append(VERSION_COLORS.get(v, "#999"))

        x = range(len(labels))
        bars = ax.bar(x, speedups, color=colors, edgecolor="white", linewidth=0.5)

        # Add value labels on bars
        for bar, sp in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{sp:.1f}x", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")

    ax.set_ylabel("Speedup vs CPU Baseline", fontsize=12)
    ax.set_title("GPU Speedup Over Sequential CPU", fontsize=14, fontweight="bold")
    ax.axhline(y=1, color="red", linestyle="--", alpha=0.5, label="CPU baseline")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "speedup.png"), dpi=150)
    plt.close()
    print(f"  → speedup.png")


def plot_convergence(results, output_dir):
    """Line chart: accuracy over epochs for each version."""
    configs = sorted(set(r["config"] for r in results))

    for config in configs:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        for v in VERSION_ORDER:
            data = [r for r in results if r["version"] == v and r["config"] == config]
            if not data:
                continue
            data.sort(key=lambda x: x["epoch"])
            epochs = [d["epoch"] for d in data]
            acc = [d["accuracy"] for d in data]
            loss = [d["loss"] for d in data]
            label = VERSION_LABELS.get(v, v)
            color = VERSION_COLORS.get(v, "#999")

            ax1.plot(epochs, acc, "-o", label=label, color=color, markersize=4)
            ax2.plot(epochs, loss, "-o", label=label, color=color, markersize=4)

        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Test Accuracy (%)")
        ax1.set_title(f"Convergence — {config} config", fontweight="bold")
        ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Training Loss")
        ax2.set_title(f"Loss Curve — {config} config", fontweight="bold")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"convergence_{config}.png"), dpi=150)
        plt.close()
        print(f"  → convergence_{config}.png")


def plot_epoch_times(results, output_dir):
    """Bar chart: average epoch time for each version and config."""
    configs = sorted(set(r["config"] for r in results))

    fig, ax = plt.subplots(figsize=(14, 6))
    bar_width = 0.12
    x_base = range(len(configs))

    for i, v in enumerate(VERSION_ORDER):
        times = []
        for config in configs:
            data = [r for r in results if r["version"] == v and r["config"] == config]
            if data:
                times.append(sum(d["time_ms"] for d in data) / len(data))
            else:
                times.append(0)

        x = [xb + i * bar_width for xb in x_base]
        ax.bar(x, times, bar_width, label=VERSION_LABELS.get(v, v),
               color=VERSION_COLORS.get(v, "#999"), edgecolor="white")

    ax.set_xlabel("Network Configuration", fontsize=12)
    ax.set_ylabel("Average Epoch Time (ms)", fontsize=12)
    ax.set_title("Training Time Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks([xb + bar_width * len(VERSION_ORDER) / 2 for xb in x_base])
    ax.set_xticklabels(configs)
    ax.legend(fontsize=8, ncol=2)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "epoch_times.png"), dpi=150)
    plt.close()
    print(f"  → epoch_times.png")


def main():
    parser = argparse.ArgumentParser(description="Plot CudaBackProp benchmark results")
    parser.add_argument("csv_file", help="Path to benchmark CSV file")
    parser.add_argument("--output", default="benchmark/results", help="Output directory")
    args = parser.parse_args()

    if not HAS_MPL:
        print("Error: matplotlib required. Install: pip install matplotlib")
        return 1

    if not os.path.exists(args.csv_file):
        print(f"Error: File not found: {args.csv_file}")
        return 1

    results = load_csv(args.csv_file)
    print(f"Loaded {len(results)} data points from {args.csv_file}")

    os.makedirs(args.output, exist_ok=True)
    print(f"\nGenerating plots in {args.output}/")

    plot_speedup(results, args.output)
    plot_convergence(results, args.output)
    plot_epoch_times(results, args.output)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
