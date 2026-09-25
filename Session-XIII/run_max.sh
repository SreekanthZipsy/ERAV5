#!/usr/bin/env bash
# Max-batch reversible runs (with expandable segments to avoid allocator fragmentation)
# plus an exact-gradient control for midpoint_a.
set -e
cd "$(dirname "$0")"
PY=/home/ubuntu/miniconda3/envs/erav5/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$PY find_max_batch.py --mode leapfrog --h 1.0 --chunked_ce > logs/maxbatch_leapfrog_chunked.log 2>&1
B=$($PY -c "import json;m=json.load(open('artifacts/batch_sweep.json'))['leapfrog+chunkedCE']['max_batch'];print(int(0.95*m)//8*8)")
LR=$($PY -c "import math;print('%.2e'%min(1.5e-3*math.sqrt($B/16),6e-3))")
$PY train.py --mode leapfrog --h 1.0 --batch $B --lr $LR --ce_chunks $((B/8)) --warmup_frac 0.1 \
  --eval_every 25 --log_every 5 --name leapfrog_maxb$B > logs/leapfrog_maxb$B.log 2>&1

rm -rf runs/midpoint_a_maxb456
$PY train.py --mode midpoint_a --h 1.0 --batch 456 --lr 6.00e-03 --ce_chunks 57 --warmup_frac 0.1 \
  --eval_every 25 --log_every 5 --name midpoint_a_maxb456 > logs/midpoint_a_maxb456.log 2>&1

$PY train.py --mode midpoint_a --h 1.0 --batch 16 --no_rev_backward --name midpoint_a_b16_autograd \
  > logs/midpoint_a_b16_autograd.log 2>&1

$PY make_report.py > logs/make_report.log 2>&1
echo RUN_MAX_DONE
