#!/usr/bin/env python3
"""Run all benchmarks and generate plots for README."""

import subprocess
import os
import re
import csv
import sys

VERSIONS = [
    ("v0_cpu",    "V0: CPU Baseline"),
    ("v1_naive",  "V1: cuBLAS"),
    ("v2_tiled",  "V2: Tiled GEMM"),
    ("v3_fused",  "V3: Fused Kernels"),
    ("v4_streams","V4: Streams+Pinned"),
    ("v5_mixed",  "V5: Mixed Precision"),
    ("v6_tensor", "V6: Tensor Cores"),
]

CONFIGS = {
    "small":  {"arch": "784,128,10",         "label": "784->128->10"},
    "medium": {"arch": "784,512,256,10",     "label": "784->512->256->10"},
}

ACTIVATIONS = ["sigmoid", "relu"]
EPOCHS = 10
BUILD_DIR = os.path.join("build", "Release")
DATA_DIR = "data"
EXE_EXT = ".exe" if sys.platform == "win32" else ""

def run_one(exe, arch, epochs, activation):
    cmd = [exe, "--arch", arch, "--epochs", str(epochs),
           "--activation", activation, "--data", DATA_DIR, "--batch", "128"]
    print(f"  CMD: {os.path.basename(exe)} --arch {arch} --activation {activation}")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return r.stdout + r.stderr
    except Exception as e:
        return f"ERROR: {e}"

def parse(output):
    epochs = []
    for line in output.split("\n"):
        m = re.search(
            r"Epoch\s+(\d+)/\d+.*?Loss:\s*([\d.]+).*?Accuracy:\s*([\d.]+)%.*?(\d+\.\d+)\s*ms\s*$",
            line)
        if m:
            epochs.append({
                "epoch": int(m.group(1)), "loss": float(m.group(2)),
                "accuracy": float(m.group(3)), "time_ms": float(m.group(4))
            })
    total_m = re.search(r"Total training time:\s*([\d.]+)\s*ms", output)
    avg_m = re.search(r"Average epoch time:\s*([\d.]+)\s*ms", output)
    return epochs, {
        "total_ms": float(total_m.group(1)) if total_m else 0,
        "avg_ms": float(avg_m.group(1)) if avg_m else 0,
    }

def main():
    results = []
    for act in ACTIVATIONS:
        for cfg_name, cfg in CONFIGS.items():
            print(f"\n=== {cfg_name} ({cfg['label']}) | {act} ===")
            for exe_name, desc in VERSIONS:
                exe = os.path.join(BUILD_DIR, exe_name + EXE_EXT)
                if not os.path.exists(exe):
                    print(f"  SKIP {desc}"); continue

                # Skip CPU baseline on medium config (too slow for 10 epochs)
                if exe_name == "v0_cpu" and cfg_name == "medium":
                    # Run only 3 epochs for CPU medium
                    out = run_one(exe, cfg["arch"], 3, act)
                else:
                    out = run_one(exe, cfg["arch"], EPOCHS, act)

                ep_data, summary = parse(out)
                if not ep_data:
                    print(f"  FAILED: {out[:200]}"); continue

                for e in ep_data:
                    results.append({
                        "version": exe_name, "desc": desc, "config": cfg_name,
                        "arch": cfg["label"], "activation": act,
                        "epoch": e["epoch"], "loss": e["loss"],
                        "accuracy": e["accuracy"], "time_ms": e["time_ms"]
                    })
                print(f"  {desc}: {ep_data[-1]['accuracy']:.2f}% | {summary['avg_ms']:.1f} ms/epoch")

    # Write CSV
    os.makedirs("benchmark/results", exist_ok=True)
    csv_path = "benchmark/results/full_benchmark.csv"
    fields = ["version","desc","config","arch","activation","epoch","loss","accuracy","time_ms"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved {len(results)} rows to {csv_path}")

if __name__ == "__main__":
    main()
