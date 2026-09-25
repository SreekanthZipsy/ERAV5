"""Train the ~21M GPT on a fixed token budget and record loss, throughput, and memory.

Examples:
  python train.py --mode baseline --batch 16 --name baseline_b16
  python train.py --mode midpoint --batch 16 --h 0.5 --name midpoint_b16
  python train.py --mode midpoint --batch 64 --probe 4     # memory/speed probe only
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

from model import GPT, GPTConfig, MODES

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RUNS = os.path.join(HERE, "runs")


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=MODES, default="baseline")
    ap.add_argument("--name", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--n_layer", type=int, default=10)
    ap.add_argument("--tokens", type=float, default=50e6)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--min_lr_frac", type=float, default=0.1)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--h", type=float, default=0.5)
    ap.add_argument("--ce_chunks", type=int, default=1)
    ap.add_argument("--no_rev_backward", action="store_true")
    ap.add_argument("--float_state", action="store_true", help="float instead of fixed-point reversible state")
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe", type=int, default=0, help="run N steps, print memory/speed, exit")
    return ap.parse_args()


class Data:
    def __init__(self, path, T, seed):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.T = T
        self.n_seq = (len(self.tokens) - 1) // T
        self.perm = np.random.default_rng(seed).permutation(self.n_seq)
        self.pos = 0
        self.offsets = np.arange(T + 1)

    def batch(self, B):
        if self.pos + B > self.n_seq:
            self.pos = 0
        ids = self.perm[self.pos : self.pos + B]
        self.pos += B
        rows = self.tokens[(ids[:, None] * self.T) + self.offsets].astype(np.int64)
        t = torch.from_numpy(rows).pin_memory().cuda(non_blocking=True)
        return t[:, :-1], t[:, 1:]


@torch.no_grad()
def evaluate(model, val, n_seq, B=16):
    model.eval()
    val.pos = 0
    total, count = 0.0, 0
    while count < n_seq:
        b = min(B, n_seq - count)
        x, y = val.batch(b)
        with torch.autocast("cuda", dtype=torch.float16):
            total += model(x, y).item() * b
        count += b
    model.train()
    return total / count


def lr_at(step, total, args, lr):
    warm = max(1, int(args.warmup_frac * total))
    if step < warm:
        return lr * (step + 1) / warm
    prog = (step - warm) / max(1, total - warm)
    return lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


def main():
    args = parse()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg = GPTConfig(block_size=args.seq, n_layer=args.n_layer, mode=args.mode, h=args.h,
                    rev_backward=not args.no_rev_backward, exact=not args.float_state,
                    ce_chunks=args.ce_chunks)
    model = GPT(cfg).cuda()
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
    scaler = torch.amp.GradScaler("cuda")
    train = Data(os.path.join(DATA, "train.bin"), args.seq, args.seed)
    val = Data(os.path.join(DATA, "val.bin"), args.seq, 0)
    val.perm = np.arange(val.n_seq)

    tok_per_step = args.batch * args.seq
    total_steps = args.probe or int(args.tokens // tok_per_step)

    def train_step(step):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total_steps, args, args.lr)
        x, y = train.batch(args.batch)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        return loss, gnorm

    if args.probe:
        try:
            torch.cuda.reset_peak_memory_stats()
            times = []
            for s in range(args.probe):
                torch.cuda.synchronize()
                t0 = time.time()
                loss, _ = train_step(s)
                loss.item()
                torch.cuda.synchronize()
                times.append(time.time() - t0)
            dt = float(np.median(times[1:])) if len(times) > 1 else times[0]
            print("PROBE " + json.dumps({
                "mode": args.mode, "batch": args.batch, "ce_chunks": args.ce_chunks,
                "n_layer": args.n_layer, "params_m": model.num_params() / 1e6,
                "peak_alloc_gb": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
                "step_s": dt, "tok_s": tok_per_step / dt, "loss": loss.item()}))
        except torch.OutOfMemoryError:
            print("PROBE_OOM")
            sys.exit(3)
        return

    name = args.name or f"{args.mode}_b{args.batch}"
    out_dir = os.path.join(RUNS, name)
    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, "log.jsonl"), "w")
    print(f"[{name}] params={model.num_params()/1e6:.2f}M (non-emb {model.num_params(True)/1e6:.2f}M) "
          f"steps={total_steps} tok/step={tok_per_step} lr={args.lr} h={args.h}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    train_time, t_mark, tokens_seen = 0.0, None, 0
    recent, status, recon = [], "ok", []
    torch.cuda.synchronize()
    t_last = time.time()
    for step in range(total_steps):
        track = args.mode != "baseline" and not args.no_rev_backward and step % args.eval_every == 0
        model.track_recon = track
        loss, gnorm = train_step(step)
        tokens_seen += tok_per_step
        if track and model.last_recon_err is not None:
            recon.append((step, model.last_recon_err))
        if (step + 1) % args.log_every == 0 or step == total_steps - 1:
            lv = loss.item()
            torch.cuda.synchronize()
            now = time.time()
            train_time += now - t_last
            if step + 1 >= 50 and t_mark is None:
                t_mark = (train_time, tokens_seen)
            recent.append(lv)
            rec = {"step": step + 1, "tokens": tokens_seen, "loss": lv, "gnorm": gnorm.item(),
                   "lr": opt.param_groups[0]["lr"], "tok_s": tokens_seen / train_time,
                   "scale": scaler.get_scale(), "time": train_time}
            if not math.isfinite(lv):
                status = "diverged"
            if (step + 1) % args.eval_every == 0 or step == total_steps - 1 or status != "ok":
                rec["val"] = evaluate(model, val, 64)
                if recon:
                    rec["recon_err"] = recon[-1][1]
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"[{name}] step {step+1}/{total_steps} loss {lv:.4f} "
                  f"{'val %.4f ' % rec['val'] if 'val' in rec else ''}"
                  f"tok/s {rec['tok_s']:.0f} mem {torch.cuda.max_memory_allocated()/2**30:.2f}GB", flush=True)
            if status != "ok":
                break
            torch.cuda.synchronize()
            t_last = time.time()

    steady = None
    if t_mark is not None and tokens_seen > t_mark[1]:
        steady = (tokens_seen - t_mark[1]) / (train_time - t_mark[0])
    final_val = evaluate(model, val, val.n_seq) if status == "ok" else float("nan")
    tail = recent[-max(1, len(recent) // 10):]
    summary = {
        "name": name, "mode": args.mode, "batch": args.batch, "seq": args.seq, "h": args.h,
        "lr": args.lr, "ce_chunks": args.ce_chunks, "rev_backward": not args.no_rev_backward,
        "exact_state": model.exact if args.mode != "baseline" else None,
        "params_m": model.num_params() / 1e6, "steps": step + 1, "tokens": tokens_seen,
        "status": status, "final_train_loss": float(np.mean(tail)), "final_val_loss": final_val,
        "tok_s_avg": tokens_seen / train_time, "tok_s_steady": steady, "train_time_s": train_time,
        "peak_alloc_gb": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
        "recon_err": recon,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{name}] DONE " + json.dumps({k: v for k, v in summary.items() if k != "recon_err"}), flush=True)


if __name__ == "__main__":
    main()
