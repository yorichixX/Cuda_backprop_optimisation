#!/usr/bin/env python3
"""
Benchmark runner for all CudaBackProp versions.

Builds the project, runs each version with multiple configurations,
and collects results into CSV files for plotting.

Usage:
    python run_benchmark.py [--build-dir build] [--data-dir data]
                            [--epochs 10] [--configs small,medium,large]
"""

import subprocess
import sys
import os
import re
import csv
import argparse
from datetime import datetime

# Network configurations to benchmark
CONFIGS = {
    "small":  "784,128,10",
    "medium": "784,512,256,10",
    "large":  "784,1024,512,256,10",
}

# Versions to benchmark (executable names)
VERSIONS = [
    ("v0_cpu",    "V0: CPU Baseline"),
    ("v1_naive",  "V1: cuBLAS + Custom Kernels"),
    ("v2_tiled",  "V2: Tiled GEMM (Shared Memory)"),
    ("v3_fused",  "V3: Fused Kernels"),
    ("v4_streams","V4: Streams + Pinned Memory"),
    ("v5_mixed",  "V5: Mixed Precision (FP16/FP32)"),
    ("v6_tensor", "V6: Tensor Cores (WMMA)"),
]


def parse_output(output):
    """Parse training output to extract per-epoch results."""
    results = []
    for line in output.split("\n"):
        # Match: Epoch  1/10 | Loss: 0.012345 | Accuracy:  95.23% | ...Time:   123.4 ms
        m = re.search(
            r"Epoch\s+(\d+)/\d+\s*\|\s*Loss:\s*([\d.]+)\s*\|\s*Accuracy:\s*([\d.]+)%\s*\|.*?(\d+\.\d+)\s*ms",
            line
        )
        if m:
            results.append({
                "epoch": int(m.group(1)),
                "loss": float(m.group(2)),
                "accuracy": float(m.group(3)),
                "time_ms": float(m.group(4)),
            })

    # Extract total/average time
    total_m = re.search(r"Total training time:\s*([\d.]+)\s*ms", output)
    avg_m   = re.search(r"Average epoch time:\s*([\d.]+)\s*ms", output)

    summary = {
        "total_time_ms": float(total_m.group(1)) if total_m else 0,
        "avg_epoch_ms":  float(avg_m.group(1))   if avg_m   else 0,
    }

    return results, summary


def run_version(exe_path, arch, epochs, data_dir, activation="sigmoid"):
    """Run one version with given configuration. Returns (stdout, returncode)."""
    cmd = [
        exe_path,
        "--arch", arch,
        "--epochs", str(epochs),
        "--activation", activation,
        "--data", data_dir,
        "--batch", "128",
    ]
    print(f"  Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )
        return result.stdout + result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "TIMEOUT", -1
    except FileNotFoundError:
        return "NOT FOUND", -1


def main():
    parser = argparse.ArgumentParser(description="CudaBackProp Benchmark Runner")
    parser.add_argument("--build-dir", default="build", help="CMake build directory")
    parser.add_argument("--data-dir", default="data", help="MNIST data directory")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--configs", default="small,medium,large",
                        help="Comma-separated config names to run")
    parser.add_argument("--activation", default="sigmoid",
                        help="Activation function (sigmoid or relu)")
    args = parser.parse_args()

    configs_to_run = args.configs.split(",")

    # Determine executable extension
    exe_ext = ".exe" if sys.platform == "win32" else ""
    build_cfg = "Release"  # CMake config

    # Find executables
    exe_dirs = [
        os.path.join(args.build_dir, build_cfg),  # Multi-config generators (MSVC)
        args.build_dir,                            # Single-config generators (make)
    ]

    results_dir = os.path.join("benchmark", "results")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(results_dir, f"benchmark_{timestamp}.csv")

    all_results = []

    print(f"\n{'='*72}")
    print(f"  CudaBackProp Benchmark — {timestamp}")
    print(f"{'='*72}\n")

    for config_name in configs_to_run:
        if config_name not in CONFIGS:
            print(f"Unknown config: {config_name}, skipping")
            continue

        arch = CONFIGS[config_name]
        print(f"\n--- Configuration: {config_name} ({arch}) ---\n")

        for exe_name, desc in VERSIONS:
            # Find executable
            exe_path = None
            for d in exe_dirs:
                candidate = os.path.join(d, exe_name + exe_ext)
                if os.path.exists(candidate):
                    exe_path = candidate
                    break

            if exe_path is None:
                print(f"  SKIP {desc} — executable not found")
                continue

            print(f"\n  {desc}")
            output, rc = run_version(exe_path, arch, args.epochs,
                                     args.data_dir, args.activation)

            if rc != 0:
                print(f"  FAILED (rc={rc})")
                print(f"  Output: {output[:500]}")
                continue

            epochs_data, summary = parse_output(output)

            if not epochs_data:
                print(f"  WARNING: No epoch data parsed from output")
                print(f"  Output: {output[:500]}")
                continue

            # Store results
            for ep in epochs_data:
                all_results.append({
                    "version": exe_name,
                    "config": config_name,
                    "arch": arch,
                    "activation": args.activation,
                    "epoch": ep["epoch"],
                    "loss": ep["loss"],
                    "accuracy": ep["accuracy"],
                    "time_ms": ep["time_ms"],
                })

            final_acc = epochs_data[-1]["accuracy"]
            print(f"  → Final accuracy: {final_acc:.2f}%")
            print(f"  → Avg epoch time: {summary['avg_epoch_ms']:.1f} ms")

    # Write CSV
    if all_results:
        fieldnames = ["version", "config", "arch", "activation",
                      "epoch", "loss", "accuracy", "time_ms"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n\nResults saved to: {csv_path}")
    else:
        print("\nNo results collected.")

    # Print summary table
    print(f"\n{'='*72}")
    print(f"  Summary (Final Epoch)")
    print(f"{'='*72}")
    print(f"{'Version':<25} {'Config':<10} {'Accuracy':>10} {'Time (ms)':>12}")
    print(f"{'-'*25} {'-'*10} {'-'*10} {'-'*12}")

    for config_name in configs_to_run:
        for exe_name, desc in VERSIONS:
            matching = [r for r in all_results
                        if r["version"] == exe_name and r["config"] == config_name]
            if matching:
                last = matching[-1]
                print(f"{desc:<25} {config_name:<10} {last['accuracy']:>9.2f}% {last['time_ms']:>11.1f}")

    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
