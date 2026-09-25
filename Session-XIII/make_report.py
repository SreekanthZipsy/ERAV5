"""Build figures and a markdown summary table from runs/ and artifacts/."""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
ART = os.path.join(HERE, "artifacts")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

COLORS = {"baseline": "black", "midpoint": "tab:blue", "midpoint_a": "tab:cyan",
          "leapfrog": "tab:orange", "euler": "tab:red"}


def load_run(name):
    d = os.path.join(RUNS, name)
    if not os.path.exists(os.path.join(d, "summary.json")):
        return None
    s = json.load(open(os.path.join(d, "summary.json")))
    log = [json.loads(l) for l in open(os.path.join(d, "log.jsonl"))]
    return s, log


def all_runs(prefix=""):
    out = {}
    for n in sorted(os.listdir(RUNS)):
        if n.startswith(prefix):
            r = load_run(n)
            if r:
                out[n] = r
    return out


def plot_curves(runs, fname, title, ymax=6.5):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, (s, log) in runs.items():
        c = COLORS.get(s["mode"], None)
        ls = "--" if s["batch"] > 16 else "-"
        label = f"{name} (val {s['final_val_loss']:.3f})"
        axes[0].plot([r["tokens"] / 1e6 for r in log], [r["loss"] for r in log], color=c, ls=ls, lw=1, label=label)
        v = [r for r in log if "val" in r]
        axes[1].plot([r["tokens"] / 1e6 for r in v], [r["val"] for r in v], color=c, ls=ls, marker="o", ms=3, label=label)
    for ax, t in zip(axes, ("train loss", "val loss (64 seqs during training)")):
        ax.set_xlabel("tokens (M)")
        ax.set_ylabel(t)
        ax.set_ylim(3.3, ymax)
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=130)
    plt.close(fig)


def plot_h_sweep():
    runs = all_runs("sweep_")
    if not runs:
        return
    names = list(runs)
    vals = [runs[n][0]["final_val_loss"] for n in names]
    cols = [COLORS.get(runs[n][0]["mode"]) for n in names]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(names)), vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("sweep_", "") for n in names], rotation=30, ha="right")
    lo = min(v for v in vals if v == v)
    ax.set_ylim(lo - 0.2, max(v for v in vals if v == v) + 0.1)
    ax.set_ylabel("val loss after 5M tokens")
    ax.set_title("Step-size sweep (batch 16, 5M tokens)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "h_sweep.png"), dpi=130)
    plt.close(fig)


def plot_batch_sweep():
    p = os.path.join(ART, "batch_sweep.json")
    if not os.path.exists(p):
        return
    sw = json.load(open(p))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for key, d in sw.items():
        mode = key.split("+")[0]
        ls = "--" if "chunked" in key else "-"
        bs = [r["batch"] for r in d["probes"]]
        axes[0].plot(bs, [r["peak_reserved_gb"] for r in d["probes"]], marker="o", ls=ls,
                     color=COLORS.get(mode), label=f"{key} (max {d['max_batch']})")
        axes[1].plot(bs, [r["tok_s"] / 1e3 for r in d["probes"]], marker="o", ls=ls, color=COLORS.get(mode), label=key)
    axes[0].axhline(14.56, color="gray", ls=":", label="T4 capacity (14.56 GiB)")
    axes[0].set_ylabel("peak reserved memory (GB)")
    axes[1].set_ylabel("throughput (k tokens/s)")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("batch size (sequences of 1024 tokens)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Memory and throughput vs batch size on a T4")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "batch_sweep.png"), dpi=130)
    plt.close(fig)


def plot_depth():
    p = os.path.join(ART, "depth_probe.json")
    if not os.path.exists(p):
        return
    rows = json.load(open(p))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for mode in ("baseline", "midpoint"):
        rr = [r for r in rows if r["mode"] == mode and not r["oom"]]
        ax.plot([r["n_layer"] for r in rr], [r["peak_alloc_gb"] for r in rr], marker="o",
                color=COLORS[mode], label=mode)
        oom = [r["n_layer"] for r in rows if r["mode"] == mode and r["oom"]]
        for L in oom:
            ax.axvline(L, color=COLORS[mode], ls=":", alpha=0.6)
            ax.text(L, 1, f" {mode} OOM", rotation=90, color=COLORS[mode], fontsize=8)
    ax.set_xlabel("number of layers (d=384, batch 16 x 1024)")
    ax.set_ylabel("peak allocated memory (GB)")
    ax.set_title("Training memory vs depth")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "memory_vs_depth.png"), dpi=130)
    plt.close(fig)


def plot_recon():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for prefix, ls, tag in (("sweepfloat_", "--", "float state"), ("sweep_", "-", "fixed-point state")):
        for name, (s, _) in all_runs(prefix).items():
            if s["mode"] in ("baseline", "midpoint_a") or not s["recon_err"]:
                continue
            steps = [r[0] for r in s["recon_err"]]
            errs = [max(r[1], 1e-12) for r in s["recon_err"]]
            ax.plot(steps, errs, ls=ls, marker="o", color=COLORS[s["mode"]],
                    label=f"{name.replace(prefix, '')} ({tag})")
    ax.set_yscale("log")
    ax.set_xlabel("training step (batch 16)")
    ax.set_ylabel("|x_rebuilt - x| / max|x|  after reversing 10 layers")
    ax.set_title("Reconstruction error of the embedding input under fp16 autocast")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "recon_error.png"), dpi=130)
    plt.close(fig)


def table():
    rows = []
    for name, (s, _) in all_runs().items():
        if name.startswith("sweep"):
            continue
        rows.append(s)
    lines = ["| run | mode | h | batch | tok/step | lr | final train loss | final val loss | tok/s (steady) | peak alloc GB | peak reserved GB | time (min) |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in rows:
        h = "-" if s["mode"] in ("baseline", "euler") else s["h"]
        lines.append(f"| {s['name']} | {s['mode']} | {h} | {s['batch']} | {s['batch']*s['seq']} | {s['lr']:.2g} | "
                     f"{s['final_train_loss']:.4f} | {s['final_val_loss']:.4f} | {s['tok_s_steady'] or s['tok_s_avg']:.0f} | "
                     f"{s['peak_alloc_gb']:.2f} | {s['peak_reserved_gb']:.2f} | {s['train_time_s']/60:.1f} |")
    sweep = ["| sweep run | state | val loss @5M | tok/s | recon err @ step 300 |", "|---|---|---:|---:|---:|"]
    for prefix, tag in (("sweep_", "fixed-point"), ("sweepfloat_", "float")):
        for name, (s, _) in all_runs(prefix).items():
            st = "float" if s["mode"] == "midpoint_a" else tag
            rc = f"{s['recon_err'][-1][1]:.1e}" if s["recon_err"] else "-"
            sweep.append(f"| {name.replace(prefix, '')} | {st} | {s['final_val_loss']:.4f} | "
                         f"{s['tok_s_steady'] or s['tok_s_avg']:.0f} | {rc} |")
    md = "\n".join(lines) + "\n\n" + "\n".join(sweep) + "\n"
    open(os.path.join(ART, "summary.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    full = {n: r for n, r in all_runs().items() if not n.startswith("sweep")}
    b16 = {n: r for n, r in full.items() if r[0]["batch"] == 16}
    if b16:
        plot_curves(b16, "loss_b16.png", "50M tokens, batch 16 x 1024: baseline vs reversible variants")
    if full:
        plot_curves(full, "loss_all.png", "All 50M-token runs (dashed = larger batch)", ymax=8.0)
    plot_h_sweep()
    plot_batch_sweep()
    plot_depth()
    plot_recon()
    table()
