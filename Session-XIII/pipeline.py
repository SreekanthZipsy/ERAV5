"""Run the main experiments after sweep_h.sh has finished.

1. Pick the best h per reversible variant from the 5M-token sweep.
2. Full 50M-token runs at batch 16 for every reversible variant.
3. Max-batch search (with and without chunked cross-entropy) for baseline and the best variant.
4. Memory-vs-depth probe.
5. 50M-token runs at the max batch for the best variant and for the baseline.
"""

import glob
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
BASE_LR, BASE_B = 1.5e-3, 16
CE_CHUNK_SEQS = 8


def sh(args, log):
    print(">>", " ".join(args), flush=True)
    with open(os.path.join(HERE, "logs", log), "w") as f:
        subprocess.run([PY] + args, cwd=HERE, stdout=f, stderr=subprocess.STDOUT, check=False)


def summary(name):
    p = os.path.join(HERE, "runs", name, "summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    best_h = {}
    for mode in ("midpoint", "midpoint_a", "leapfrog"):
        cands = [json.load(open(f)) for f in glob.glob(os.path.join(HERE, f"runs/sweep_{mode}_h*/summary.json"))]
        cands = [c for c in cands if c["status"] == "ok" and math.isfinite(c["final_val_loss"])]
        best_h[mode] = min(cands, key=lambda c: c["final_val_loss"])["h"] if cands else 0.5
    print("best h:", best_h, flush=True)

    for mode in ("midpoint", "euler", "midpoint_a", "leapfrog"):
        name = f"{mode}_b16"
        if summary(name) is None:
            sh(["train.py", "--mode", mode, "--batch", "16", "--h", str(best_h.get(mode, 0.5)), "--name", name],
               f"{name}.log")

    rev = [summary(f"{m}_b16") for m in ("midpoint", "euler", "midpoint_a", "leapfrog")]
    rev = [s for s in rev if s and s["status"] == "ok"]
    best = min(rev, key=lambda s: s["final_val_loss"])
    print("best reversible variant:", best["mode"], best["final_val_loss"], flush=True)
    json.dump({"best_h": best_h, "best_mode": best["mode"]},
              open(os.path.join(HERE, "artifacts", "choices.json"), "w"), indent=2)

    hb = str(best["h"])
    for mode, extra in (("baseline", []), ("baseline", ["--chunked_ce"]),
                        (best["mode"], ["--h", hb]), (best["mode"], ["--h", hb, "--chunked_ce"])):
        tag = mode + ("_chunked" if extra and extra[-1] == "--chunked_ce" else "")
        sh(["find_max_batch.py", "--mode", mode] + extra, f"maxbatch_{tag}.log")

    sh(["depth_probe.py"], "depth_probe.log")

    sw = json.load(open(os.path.join(HERE, "artifacts", "batch_sweep.json")))
    for key, mode, h in ((f"{best['mode']}+chunkedCE", best["mode"], hb), ("baseline", "baseline", "0.5")):
        maxb = sw[key]["max_batch"]
        b = int(0.95 * maxb) // 8 * 8 if maxb >= 16 else maxb
        lr = min(BASE_LR * math.sqrt(b / BASE_B), 6e-3)
        ce = max(1, b // CE_CHUNK_SEQS) if "chunked" in key else 1
        name = f"{mode}_maxb{b}"
        sh(["train.py", "--mode", mode, "--batch", str(b), "--h", h, "--lr", f"{lr:.2e}",
            "--ce_chunks", str(ce), "--warmup_frac", "0.1", "--eval_every", "25", "--log_every", "5",
            "--name", name], f"{name}.log")

    sh(["make_report.py"], "make_report.log")
    print("PIPELINE DONE", flush=True)


if __name__ == "__main__":
    main()
