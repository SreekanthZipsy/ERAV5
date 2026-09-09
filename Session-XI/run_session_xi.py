#!/usr/bin/env python3
"""Run Session-XI lab end-to-end; write artifacts + figures."""
from __future__ import annotations

import json
from pathlib import Path

from adam_lab import (
    AdamHyper,
    adam_five_grads_vs_pytorch,
    compare_bias_correction_trajectory,
    lr_sweep_width,
    train_log_update_ratios,
    train_schedule,
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

    # ------------------------------------------------------------------
    # 1) Adam by hand vs PyTorch
    # ------------------------------------------------------------------
    banner("1) Adam by hand — one weight, five gradients")
    grads = [1.0, -0.5, 0.25, 2.0, -1.5]
    w0 = 0.5
    hp = AdamHyper(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8)
    cmp_ = adam_five_grads_vs_pytorch(w0=w0, grads=grads, hp=hp)
    print(f"w0={w0}  grads={grads}")
    print(f"{'t':>2} {'g':>6} | {'m':>12} {'v':>12} {'m̂':>12} {'v̂':>12} {'step':>12} {'w':>12}")
    for row in cmp_["hand_with_bias_corr"]:
        print(
            f"{row['t']:2d} {row['g']:6.2f} | "
            f"{row['m']:12.8f} {row['v']:12.8f} {row['m_hat']:12.8f} "
            f"{row['v_hat']:12.8f} {row['step']:12.8f} {row['w']:12.8f}"
        )
    print("\nMax |hand − PyTorch| over m,v,m̂,v̂,step,w:", f"{cmp_['max_abs_diff']:.2e}")
    for d in cmp_["diffs"]:
        print(
            f"  t={d['t']}: dm={d['dm']:.2e} dv={d['dv']:.2e} "
            f"dm̂={d['dm_hat']:.2e} dv̂={d['dv_hat']:.2e} dstep={d['dstep']:.2e} dw={d['dw']:.2e}"
        )
    assert cmp_["max_abs_diff"] < 1e-10, "Adam hand vs PyTorch mismatch"
    results["adam_hand"] = {
        "w0": w0,
        "grads": grads,
        "hand": cmp_["hand_with_bias_corr"],
        "pytorch": cmp_["pytorch"],
        "max_abs_diff": cmp_["max_abs_diff"],
    }

    # ------------------------------------------------------------------
    # 2) Bias correction on/off — first 20 steps
    # ------------------------------------------------------------------
    banner("2) Bias correction on vs off — first 20 steps")
    traj = compare_bias_correction_trajectory(
        w0=0.5, grads=grads, steps=20, hp=AdamHyper(lr=1e-2), horizon=5000
    )
    print(f"Rel |Δstep| at t=20 (cycling grads): {traj['rel_diff_at_20']:.3f}")
    print(f"β1 reaches ~99% of asymptote around t≈{traj['t_beta1_99pct']}")
    print(f"β2 reaches ~99% of asymptote around t≈{traj['t_beta2_99pct']}")
    print(
        f"Steps until |Δstep| stays <1% of |step_bc| (const g): "
        f"{traj['steps_until_diff_immaterial']}"
    )
    print(f"Criterion: {traj['criterion']}")
    for t in range(1, 6):
        a = traj["with_bias_correction"][t - 1]
        b = traj["without_bias_correction"][t - 1]
        print(
            f"  t={t}: step_bc={a['step']:.6f}  step_raw={b['step']:.6f}  "
            f"|Δ|={traj['step_abs_diff'][t-1]:.6f}"
        )
    results["bias_correction"] = {
        "steps_until_immaterial": traj["steps_until_diff_immaterial"],
        "rel_diff_at_20": traj["rel_diff_at_20"],
        "t_beta1_99pct": traj["t_beta1_99pct"],
        "t_beta2_99pct": traj["t_beta2_99pct"],
        "step_abs_diff": traj["step_abs_diff"],
        "with_bc_steps": [r["step"] for r in traj["with_bias_correction"]],
        "without_bc_steps": [r["step"] for r in traj["without_bias_correction"]],
        "criterion": traj["criterion"],
    }

    try:
        import matplotlib.pyplot as plt

        ts = list(range(1, 21))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ts, results["bias_correction"]["with_bc_steps"], label="with bias correction")
        ax.plot(
            ts,
            results["bias_correction"]["without_bc_steps"],
            label="without (m,v raw)",
            linestyle="--",
        )
        ax.set_xlabel("step")
        ax.set_ylabel("Adam update magnitude")
        ax.set_title("Bias correction — first 20 steps (still matters a lot)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG / "bias_correction.png", dpi=140)
        plt.close(fig)
        print(f"wrote {FIG / 'bias_correction.png'}")
    except Exception as e:
        print("plot skipped:", e)

    # ------------------------------------------------------------------
    # 3) Update-to-weight ratio + warmup
    # ------------------------------------------------------------------
    banner("3) Update-to-weight ratio; when warmup stops changing it")
    ur = train_log_update_ratios(steps=80, warmup=20, lr_max=3e-3, seed=0)
    print(f"scheduled warmup steps: {ur['warmup_steps_scheduled']}")
    print(f"warmup stops changing ||ΔW||/||W|| around step: {ur['step_warmup_stops_changing_ratio']}")
    print(ur["note"])
    for s in [0, 5, 10, 19, 20, 21, 40]:
        print(
            f"  step {s:2d}: lr={ur['hist']['lr'][s]:.5f}  "
            f"median_ratio={ur['hist']['ratio_median'][s]:.6e}  "
            f"mean_ratio={ur['hist']['ratio_mean'][s]:.6e}"
        )
    # sample per-layer at warmup end
    sample = ur["hist"]["ratios"][ur["step_warmup_stops_changing_ratio"]]
    print("per-layer ratios at that step (first 8):")
    for i, (k, v) in enumerate(sample.items()):
        if i >= 8:
            break
        print(f"    {k}: {v:.6e}")
    results["update_ratio"] = {
        "warmup_scheduled": ur["warmup_steps_scheduled"],
        "step_warmup_stops_changing_ratio": ur["step_warmup_stops_changing_ratio"],
        "lr": ur["hist"]["lr"],
        "ratio_mean": ur["hist"]["ratio_mean"],
        "ratio_median": ur["hist"]["ratio_median"],
        "loss": ur["hist"]["loss"],
        "ratios_at_key_step": sample,
        "note": ur["note"],
    }

    try:
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(
            ur["hist"]["step"],
            ur["hist"]["ratio_median"],
            color="#0d5c63",
            label="median ||ΔW||/||W||",
        )
        ax1.axvline(ur["warmup_steps_scheduled"], color="#888", ls="--", label="warmup end (LR)")
        ax1.axvline(
            ur["step_warmup_stops_changing_ratio"],
            color="#c45c26",
            ls=":",
            label="warmup stops driving ratio",
        )
        ax1.set_ylabel("update / weight")
        ax2 = ax1.twinx()
        ax2.plot(ur["hist"]["step"], ur["hist"]["lr"], color="#b8860b", alpha=0.7, label="lr")
        ax2.set_ylabel("lr")
        ax1.set_title("Update-to-weight ratio vs warmup")
        ax1.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(FIG / "update_ratio_warmup.png", dpi=140)
        plt.close(fig)
        print(f"wrote {FIG / 'update_ratio_warmup.png'}")
    except Exception as e:
        print("plot skipped:", e)

    # ------------------------------------------------------------------
    # 4) Cosine vs WSD — 300 steps, report at 200
    # ------------------------------------------------------------------
    banner("4) Cosine vs WSD — train 300, stop-report at 200")
    cos = train_schedule("cosine", total_steps=300, stop_at=200, lr_max=3e-3, warmup=30, seed=1)
    wsd = train_schedule("wsd", total_steps=300, stop_at=200, lr_max=3e-3, warmup=30, seed=1)
    print(
        f"COSINE @200: loss={cos['loss_at_step_200']:.4f}  "
        f"mean10={cos['mean_loss_last10_at_200']:.4f}"
    )
    print(
        f"WSD    @200: loss={wsd['loss_at_step_200']:.4f}  "
        f"mean10={wsd['mean_loss_last10_at_200']:.4f}"
    )
    print(
        f"COSINE @300: {cos['loss_at_300']:.4f} (mean10={cos['mean_loss_last10_at_300']:.4f})  "
        f"WSD @300: {wsd['loss_at_300']:.4f} (mean10={wsd['mean_loss_last10_at_300']:.4f})"
    )
    keep = (
        "cosine"
        if cos["mean_loss_last10_at_200"] <= wsd["mean_loss_last10_at_200"]
        else "wsd"
    )
    keep_reason = (
        f"At step 200 (WSD still in stable phase; decay starts at {wsd['decay_start']}), "
        f"mean-last-10 loss is cosine={cos['mean_loss_last10_at_200']:.4f} vs "
        f"WSD={wsd['mean_loss_last10_at_200']:.4f}. "
        f"I would keep **{keep}** as the mid-run checkpoint. "
        "Class takeaway: WSD often wins after the sharp drop but feels risky mid-run "
        "because loss plateaus while LR stays high."
    )
    print("Keep:", keep)
    print(keep_reason)
    results["schedules"] = {
        "cosine": {
            "loss_at_200": cos["loss_at_step_200"],
            "mean10_at_200": cos["mean_loss_last10_at_200"],
            "loss_at_300": cos["loss_at_300"],
            "mean10_at_300": cos["mean_loss_last10_at_300"],
            "losses": cos["losses"],
            "lrs": cos["lrs"],
        },
        "wsd": {
            "loss_at_200": wsd["loss_at_step_200"],
            "mean10_at_200": wsd["mean_loss_last10_at_200"],
            "loss_at_300": wsd["loss_at_300"],
            "mean10_at_300": wsd["mean_loss_last10_at_300"],
            "losses": wsd["losses"],
            "lrs": wsd["lrs"],
            "decay_start": wsd["decay_start"],
        },
        "keep_at_200": keep,
        "keep_reason": keep_reason,
    }

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(cos["losses"], label="cosine")
        ax[0].plot(wsd["losses"], label="WSD")
        ax[0].axvline(200, color="#888", ls="--")
        ax[0].set_title("Train loss")
        ax[0].legend()
        ax[0].set_xlabel("step")
        ax[1].plot(cos["lrs"], label="cosine LR")
        ax[1].plot(wsd["lrs"], label="WSD LR")
        ax[1].axvline(200, color="#888", ls="--")
        ax[1].set_title("Learning rate")
        ax[1].legend()
        ax[1].set_xlabel("step")
        fig.tight_layout()
        fig.savefig(FIG / "cosine_vs_wsd.png", dpi=140)
        plt.close(fig)
        print(f"wrote {FIG / 'cosine_vs_wsd.png'}")
    except Exception as e:
        print("plot skipped:", e)

    # ------------------------------------------------------------------
    # 5) LR sweep × width
    # ------------------------------------------------------------------
    banner("5) LR sweep at widths 256 / 512 / 1024")
    sweep = lr_sweep_width(
        widths=[256, 512, 1024],
        lrs=[1e-5, 2e-5, 3e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3, 2e-3],
        steps=100,
        seed=0,
    )
    for w, rows in sweep["grid"].items():
        print(f"width={w}:")
        for r in rows:
            mark = " <-- min" if r["lr"] == sweep["minima"][str(w)]["lr"] else ""
            print(f"  lr={r['lr']:.0e}  loss={r['loss']:.4f}{mark}")
    print(
        f"Fit η*(4096)={sweep['lr_at_4096_extrapolated']:.3e}  "
        f"inv-width={sweep['lr_at_4096_inv_width_from_1024']:.3e}  "
        f"μP-transfer={sweep['lr_at_4096_mup_transfer']:.3e}"
    )
    print(f"slope={sweep['fit_log_lr_vs_log_width']['slope']:.3f}  confidence={sweep['confidence']}")
    results["lr_sweep"] = {
        "grid": {str(k): v for k, v in sweep["grid"].items()},
        "minima": sweep["minima"],
        "lr_4096_fit": sweep["lr_at_4096_extrapolated"],
        "lr_4096_inv_width": sweep["lr_at_4096_inv_width_from_1024"],
        "lr_4096_mup": sweep["lr_at_4096_mup_transfer"],
        "fit": sweep["fit_log_lr_vs_log_width"],
        "confidence": sweep["confidence"],
        "note": sweep["note"],
    }

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        for w, rows in sweep["grid"].items():
            xs = [r["lr"] for r in rows]
            ys = [r["loss"] for r in rows]
            ax.plot(xs, ys, marker="o", label=f"width={w}")
            m = sweep["minima"][str(w)]
            ax.scatter([m["lr"]], [m["loss"]], s=80, zorder=5)
            ax.annotate(
                f"min@{m['lr']:g}",
                (m["lr"], m["loss"]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
            )
        ax.set_xscale("log")
        ax.set_xlabel("learning rate")
        ax.set_ylabel("mean loss (last 20/80 steps)")
        ax.set_title("LR sweep × width (both sides tuned)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG / "lr_sweep_width.png", dpi=140)
        plt.close(fig)
        print(f"wrote {FIG / 'lr_sweep_width.png'}")
    except Exception as e:
        print("plot skipped:", e)

    (ART / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {ART / 'results.json'}")
    banner("DONE")
    return results


if __name__ == "__main__":
    main()
