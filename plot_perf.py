#!/usr/bin/env python3
"""Plot VibeGEMM performance results for A100 and H100."""

import re
import matplotlib.pyplot as plt
import os


def parse_perf_file(filepath):
    """Parse a perf result file, return (cublas_tflops, versions[], tflops[])."""
    with open(filepath, "r") as f:
        text = f.read()

    m = re.search(r"\[cuBLAS\].*?(\d+\.\d+)\s+TFLOPS", text, re.DOTALL)
    cublas = float(m.group(1)) if m else 0.0

    versions = []
    tflops = []
    for m in re.finditer(
        r"\[VibeGEMM.*?v(\d+)\].*?(\d+\.\d+)\s+TFLOPS", text, re.DOTALL
    ):
        versions.append(f"v{m.group(1)}")
        tflops.append(float(m.group(2)))

    return cublas, versions, tflops


def plot_gpu(filepath, gpu_name, color, outpath):
    """Plot a single GPU's performance and save to outpath."""
    cublas, versions, tflops = parse_perf_file(filepath)

    x = list(range(len(versions)))

    fig, ax = plt.subplots(figsize=(10, 6))

    # VibeGEMM line
    ax.plot(x, tflops, marker="o", color=color, linewidth=2, markersize=5,
            label="VibeGEMM", zorder=3)

    # cuBLAS baseline — blue dashed line
    ax.axhline(y=cublas, color="#1f77b4", linestyle="--", linewidth=2,
               label=f"cuBLAS ({cublas:.1f} TFLOPS)")

    # Annotate peak
    peak = max(tflops)
    peak_idx = tflops.index(peak)
    pct = peak / cublas * 100
    ax.annotate(
        f"{peak:.1f} TFLOPS\n({pct:.1f}% of cuBLAS)",
        xy=(peak_idx, peak),
        xytext=(0, 14),
        textcoords="offset points",
        ha="center",
        fontsize=8,
        color=color,
        fontweight="bold",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(versions, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("VibeGEMM version")
    ax.set_ylabel("TFLOPS")
    ax.set_title(f"{gpu_name} — GEMM 8192×8192×8192 (FP16)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {outpath}")


def main():
    os.makedirs("results", exist_ok=True)

    tasks = [
        ("results/perf_a100.txt", "NVIDIA A100", "#d62728", "results/perf_a100.png"),
        ("results/perf_h100.txt", "NVIDIA H100", "#d62728", "results/perf_h100.png"),
    ]

    for path, gpu, color, out in tasks:
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping.")
            continue
        plot_gpu(path, gpu, color, out)


if __name__ == "__main__":
    main()