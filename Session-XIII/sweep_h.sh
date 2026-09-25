#!/usr/bin/env bash
# Short (5M-token) runs at batch 16 to pick the step size h for each reversible variant.
# midpoint_a always uses float state, so its earlier sweepfloat_* runs are reused.
set -e
cd "$(dirname "$0")"
PY=/home/ubuntu/miniconda3/envs/erav5/bin/python
T=5e6
run() { $PY train.py --tokens $T --batch 16 --eval_every 100 "$@" 2>&1 | grep -E "DONE|diverged" ; }

[ -f runs/sweep_baseline/summary.json ] || run --mode baseline --name sweep_baseline
for h in 0.5 1.0; do
  [ -d runs/sweep_midpoint_a_h$h ] || cp -r runs/sweepfloat_midpoint_a_h$h runs/sweep_midpoint_a_h$h
done
for h in 0.25 0.5 1.0; do run --mode midpoint --h $h --name sweep_midpoint_h$h; done
for h in 0.5 1.0;      do run --mode leapfrog --h $h --name sweep_leapfrog_h$h; done
run --mode euler --name sweep_euler
