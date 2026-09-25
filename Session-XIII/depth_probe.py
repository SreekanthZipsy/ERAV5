"""Peak training memory vs depth at a fixed batch (paper Fig. 3 analogue).

  python depth_probe.py   -> artifacts/depth_probe.json
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "artifacts", "depth_probe.json")


def main():
    batch = 16
    res = []
    for mode in ("baseline", "midpoint"):
        for L in (4, 10, 20, 40, 80):
            cmd = [sys.executable, "train.py", "--mode", mode, "--batch", str(batch), "--n_layer", str(L),
                   "--probe", "4"]
            r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
            row = {"mode": mode, "n_layer": L, "batch": batch, "oom": True}
            for line in r.stdout.splitlines():
                if line.startswith("PROBE "):
                    row = json.loads(line[6:]) | {"oom": False}
            print(row, flush=True)
            res.append(row)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
