#!/usr/bin/env python3
"""Run Session-XII ZeRO simulation; write artifacts + figures."""
from __future__ import annotations

import json
from pathlib import Path

from zero_sim import (
    WORLD_SIZE_DEFAULT,
    ZeroStage,
    compare_all_stages,
    format_bytes,
    lecture_30b_table,
    memory_for_stage,
    scale_report,
    simulate_stage,
)

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
FIG = ROOT / "figures"
ART.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)


def banner(t: str) -> None:
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


def main() -> dict:
    results: dict = {}
    world = WORLD_SIZE_DEFAULT

    # ------------------------------------------------------------------
    banner(f"1) Spin up {world} virtual GPUs + demo MLP step per ZeRO stage")
    # ------------------------------------------------------------------
    runs = compare_all_stages(world_size=world, width=256, depth=4, batch=8, seed=0)
    print(f"{'stage':<8} {'params':>10} {'per-GPU':>14} {'comm/step':>14} "
          f"{'wall_ms':>10} {'compute%':>10} {'loss':>8}")
    summary = {}
    for name, r in runs.items():
        gb = r.accounting.as_gb()
        print(
            f"{name:<8} {r.num_params:>10d} "
            f"{format_bytes(r.accounting.total_per_gpu_bytes):>14} "
            f"{format_bytes(r.accounting.comm_bytes_per_step):>14} "
            f"{r.wall_ms:>10.1f} {100*r.compute_fraction:>9.1f}% {r.mean_loss:>8.4f}"
        )
        summary[name] = {
            "num_params": r.num_params,
            "world_size": r.world_size,
            "per_gpu_bytes": r.accounting.total_per_gpu_bytes,
            "params_bytes": r.accounting.params_bytes,
            "grads_bytes": r.accounting.grads_bytes,
            "optim_bytes": r.accounting.optim_bytes,
            "comm_bytes": r.accounting.comm_bytes_per_step,
            "wall_ms": r.wall_ms,
            "compute_fraction": r.compute_fraction,
            "mean_loss": r.mean_loss,
            "notes": r.notes,
            "rank0_step_ms": r.per_rank[0].step_ms,
        }
    results["demo_32gpu"] = summary

    # ------------------------------------------------------------------
    banner("2) Lecture-scale ladder — 30B params @ 8 / 32 / 64 GPUs")
    # ------------------------------------------------------------------
    ladder = {}
    for n in (8, 32, 64):
        table = lecture_30b_table(n)
        ladder[str(n)] = {}
        print(f"\n--- world_size = {n} (30B params, BF16+Adam 16 B/param baseline) ---")
        print(f"{'stage':<8} {'per-GPU':>12} {'cluster':>12} {'comm/step':>12}")
        for s, m in table.items():
            gb = m.as_gb()
            print(
                f"{s:<8} {gb['per_gpu_GiB']:>10.1f} GiB "
                f"{gb['cluster_GiB']:>10.1f} GiB "
                f"{gb['comm_GiB_per_step']:>10.2f} GiB"
            )
            ladder[str(n)][s] = gb
    results["lecture_30b"] = ladder

    # ------------------------------------------------------------------
    banner("3) Memory & compute change plots")
    # ------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        stages = ["dp", "zero1", "zero2", "zero3"]
        labels = ["DP", "ZeRO-1", "ZeRO-2", "ZeRO-3"]
        colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]

        # --- stacked per-GPU memory for the demo model ---
        fig, ax = plt.subplots(figsize=(8, 4.5))
        params_b = [summary[s]["params_bytes"] / 1024**2 for s in stages]
        grads_b = [summary[s]["grads_bytes"] / 1024**2 for s in stages]
        optim_b = [summary[s]["optim_bytes"] / 1024**2 for s in stages]
        x = np.arange(len(stages))
        ax.bar(x, params_b, label="params (fp16)", color="#4c78a8")
        ax.bar(x, grads_b, bottom=params_b, label="grads (fp16)", color="#f58518")
        bottom2 = [a + b for a, b in zip(params_b, grads_b)]
        ax.bar(x, optim_b, bottom=bottom2, label="optim (fp32+m+v)", color="#54a24b")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("MiB per virtual GPU")
        ax.set_title(f"Resident memory on each of {world} virtual GPUs (demo MLP)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG / "memory_per_gpu_stages.png", dpi=140)
        plt.close(fig)

        # --- communication vs compute fraction ---
        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        comm = [summary[s]["comm_bytes"] / 1024**2 for s in stages]
        comp = [100 * summary[s]["compute_fraction"] for s in stages]
        ax1.bar(x - 0.2, comm, width=0.4, color="#e45756", label="comm MiB / step")
        ax1.set_ylabel("Communication (MiB / step)")
        ax2 = ax1.twinx()
        ax2.plot(x, comp, "o-", color="#4c78a8", label="compute fraction %")
        ax2.set_ylabel("Compute / (compute + comm tax) %")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.set_title("As we shard more, memory falls but communication rises")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="best")
        fig.tight_layout()
        fig.savefig(FIG / "comm_vs_compute.png", dpi=140)
        plt.close(fig)

        # --- 30B per-GPU memory across world sizes ---
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ws = [8, 32, 64]
        for s, lab, c in zip(stages, labels, colors):
            ys = [ladder[str(n)][s]["per_gpu_GiB"] for n in ws]
            ax.plot(ws, ys, "o-", label=lab, color=c)
        ax.set_xlabel("world size (# GPUs)")
        ax.set_ylabel("GiB per GPU")
        ax.set_title("30B model: per-GPU memory vs world size (lecture ladder)")
        ax.set_xticks(ws)
        ax.legend()
        ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(FIG / "memory_30b_vs_worldsize.png", dpi=140)
        plt.close(fig)

        # --- stacked for 30B @ 32 GPUs ---
        fig, ax = plt.subplots(figsize=(8, 4.5))
        n = 32
        psi = 30_000_000_000
        pb, gb_, ob = [], [], []
        for s in ZeroStage:
            m = memory_for_stage(s, psi, n)
            pb.append(m.params_bytes / 1024**3)
            gb_.append(m.grads_bytes / 1024**3)
            ob.append(m.optim_bytes / 1024**3)
        ax.bar(x, pb, label="params", color="#4c78a8")
        ax.bar(x, gb_, bottom=pb, label="grads", color="#f58518")
        b2 = [a + b for a, b in zip(pb, gb_)]
        ax.bar(x, ob, bottom=b2, label="optim", color="#54a24b")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("GiB per GPU")
        ax.set_title("30B params @ 32 GPUs — what each ZeRO stage keeps resident")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG / "memory_30b_32gpu_stacked.png", dpi=140)
        plt.close(fig)

        print(f"wrote figures under {FIG}")
    except Exception as e:
        print("plot skipped:", e)

    # ------------------------------------------------------------------
    banner("4) Scaling table export")
    # ------------------------------------------------------------------
    rows = scale_report(
        param_counts=[1_000_000_000, 7_000_000_000, 30_000_000_000],
        world_sizes=[8, 32, 64],
    )
    results["scale_rows"] = rows

    (ART / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {ART / 'results.json'}")
    banner("DONE")
    return results


if __name__ == "__main__":
    main()
