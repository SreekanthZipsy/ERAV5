"""Find the largest batch (seq=1024) that trains without OOM, and record speed vs batch.

Each probe runs `train.py --probe` in a fresh process so allocator state never leaks
between trials. Results go to artifacts/batch_sweep.json.

  python find_max_batch.py --mode baseline
  python find_max_batch.py --mode midpoint --chunked_ce
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "artifacts", "batch_sweep.json")
CE_CHUNK_SEQS = 8


def probe(mode, batch, chunked, extra):
    ce = max(1, batch // CE_CHUNK_SEQS) if chunked else 1
    cmd = [sys.executable, "train.py", "--mode", mode, "--batch", str(batch),
           "--probe", "4", "--ce_chunks", str(ce)] + extra
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("PROBE "):
            return json.loads(line[6:])
    if "PROBE_OOM" in r.stdout or "OutOfMemory" in r.stderr:
        return None
    raise RuntimeError(r.stderr[-2000:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="baseline")
    ap.add_argument("--chunked_ce", action="store_true")
    ap.add_argument("--h", type=float, default=0.5)
    ap.add_argument("--tol", type=int, default=4)
    args = ap.parse_args()
    extra = ["--h", str(args.h)]
    key = f"{args.mode}{'+chunkedCE' if args.chunked_ce else ''}"

    results = {}
    b, last_ok, first_bad = 8, None, None
    while True:
        r = probe(args.mode, b, args.chunked_ce, extra)
        print(key, b, "OOM" if r is None else f"{r['peak_reserved_gb']:.2f}GB {r['tok_s']:.0f} tok/s", flush=True)
        if r is None:
            first_bad = b
            break
        results[b] = r
        last_ok = b
        b *= 2
    lo, hi = last_ok, first_bad
    while hi - lo > args.tol:
        mid = (lo + hi) // 2
        r = probe(args.mode, mid, args.chunked_ce, extra)
        print(key, mid, "OOM" if r is None else f"{r['peak_reserved_gb']:.2f}GB {r['tok_s']:.0f} tok/s", flush=True)
        if r is None:
            hi = mid
        else:
            results[mid] = r
            lo = mid

    allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
    allres[key] = {"max_batch": lo, "first_oom": hi,
                   "probes": [results[k] for k in sorted(results)]}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(allres, f, indent=2)
    print(f"{key}: max batch {lo} (OOM at {hi})")


if __name__ == "__main__":
    main()
