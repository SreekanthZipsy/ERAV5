#!/usr/bin/env python3
"""Run Kronecker Embedding V2 proofs and emit report artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from kronecker_v2.experiments import run_all


def plot_curves(results: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    # Text LM losses
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, color in (("kronecker", "#0B6E4F"), ("table", "#C44536")):
        c = results["train_text"][key]["curve"]
        ax.plot([r["step"] for r in c], [r["loss"] for r in c], label=results["train_text"][key]["name"], color=color, lw=2)
    ax.set_xlabel("step")
    ax.set_ylabel("CE loss")
    ax.set_title("Text LM: Kronecker V2 vs learned table")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "text_lm_loss.png", dpi=140)
    plt.close()

    # Image acc
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, label, color in (
        ("kronecker", "Kronecker image", "#0B6E4F"),
        ("table_hash", "Hash table baseline", "#C44536"),
    ):
        c = results["train_image"][key]["curve"]
        ax.plot([r["step"] for r in c], [r["acc"] for r in c], label=label, color=color, lw=2)
    ax.set_xlabel("step")
    ax.set_ylabel("accuracy")
    ax.set_title("Image shape classification (8×8 → 4 patches)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "image_acc.png", dpi=140)
    plt.close()

    # Audio acc
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, label, color in (
        ("kronecker", "Kronecker audio", "#0B6E4F"),
        ("table_hash", "Hash table baseline", "#C44536"),
    ):
        c = results["train_audio"][key]["curve"]
        ax.plot([r["step"] for r in c], [r["acc"] for r in c], label=label, color=color, lw=2)
    ax.set_xlabel("step")
    ax.set_ylabel("accuracy")
    ax.set_title("Audio tone classification (μ-law frames)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "audio_acc.png", dpi=140)
    plt.close()

    # Locality bars for image
    loc = results["locality"]["image"]
    fig, ax = plt.subplots(figsize=(7, 4))
    keys = ["self", "brightness_shift", "pixel_shuffle", "different_shape"]
    vals = [loc[k] for k in keys]
    ax.bar(keys, vals, color=["#0B6E4F", "#3D8B6E", "#C44536", "#8B1E16"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("cosine(κ, ·)")
    ax.set_title("Image locality: structure preserved under brightness, not shuffle")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(out / "image_locality.png", dpi=140)
    plt.close()

    # Audio locality
    loc = results["locality"]["audio"]
    fig, ax = plt.subplots(figsize=(7, 4))
    keys = ["self", "additive_noise", "phase_shift", "different_freq"]
    vals = [loc[k] for k in keys]
    ax.bar(keys, vals, color=["#0B6E4F", "#3D8B6E", "#E09F3E", "#C44536"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("cosine(κ, ·)")
    ax.set_title("Audio locality: noise-robust, frequency-sensitive")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(out / "audio_locality.png", dpi=140)
    plt.close()

    # Param accounting
    p = results["params_frontier_scale"]
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["Text table\n|V|×d", "K text\nD×d", "K image\nD×d", "K audio\nD×d"]
    vals = [
        p["classic_text_table"] / 1e6,
        p["kronecker_text_proj"] / 1e6,
        p["kronecker_image_proj"] / 1e6,
        p["kronecker_audio_proj"] / 1e6,
    ]
    ax.bar(labels, vals, color=["#C44536", "#0B6E4F", "#2A9D8F", "#264653"])
    ax.set_ylabel("millions of parameters (d=4096)")
    ax.set_title("Frontier-scale input params (V=131072)")
    fig.tight_layout()
    fig.savefig(out / "params.png", dpi=140)
    plt.close()

    # Kronecker animation frames — sparse activation schematic
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    titles = ["Text: byte⊗pos", "Image: atom⊗row⊗col", "Audio: q⊗time"]
    for ax, title in zip(axes, titles):
        grid = np.zeros((12, 16))
        rng = np.random.default_rng(abs(hash(title)) % (2**31))
        for _ in range(6):
            r, c = int(rng.integers(0, 12)), int(rng.integers(0, 16))
            grid[r, c] = 1.0
        ax.imshow(grid, cmap="Greens", vmin=0, vmax=1, aspect="auto")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("κ is sparse: only |S| active atom×coord cells", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "sparse_schematic.png", dpi=140)
    plt.close()


from report_builder import write_report_html  # problem/solution-framed static report



def main() -> int:
    artifacts = ROOT / "artifacts"
    plots = ROOT / "report" / "plots"
    print("[EVENT] running multimodal Kronecker V2 experiments…")
    results = run_all(artifacts)
    print("[PASS] experiments_complete")
    plot_curves(results, plots)
    print("[PASS] plots_written")
    write_report_html(artifacts, plots, ROOT / "report" / "index.html")
    # copy plots path relative
    print("[PASS] report_written", ROOT / "report" / "index.html")
    print(
        json.dumps(
            {
                "text_kron_loss": results["train_text"]["kronecker"]["final_loss"],
                "text_table_loss": results["train_text"]["table"]["final_loss"],
                "image_kron_acc": results["train_image"]["kronecker"]["final_acc"],
                "audio_kron_acc": results["train_audio"]["kronecker"]["final_acc"],
                "text_param_savings": results["params_frontier_scale"]["text_savings_vs_table"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
