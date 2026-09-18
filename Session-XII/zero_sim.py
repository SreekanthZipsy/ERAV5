"""
Session XII — 32 virtual GPUs + ZeRO-1 / ZeRO-2 / ZeRO-3 simulator.

This is an *educational* simulator, not DeepSpeed/FSDP. It:
  - spins up WORLD_SIZE virtual ranks (threads)
  - runs a tiny demo MLP on each rank
  - accounts memory / communication / wall-time the way ZeRO stages do

Memory model (BF16 training + Adam, bytes per parameter Ψ):
  fp16 params ........ 2
  fp16 grads ......... 2
  fp32 master + m + v  12   (optimizer / "Adam states" in the lecture)
  ---------------------
  DP total ........... 16 bytes/param/GPU   (lecture: ~16× for one param)

ZeRO-1: shard the 12-byte optimizer block
ZeRO-2: also shard the 2-byte grads
ZeRO-3: also shard the 2-byte params  (all-gather before compute)

Comm model (bytes moved per step, order-of-magnitude, matching lecture "2P"):
  DP     : all-reduce grads ≈ 2Ψ
  ZeRO-1 : all-reduce grads + all-gather updated params ≈ 2Ψ + 2Ψ
  ZeRO-2 : reduce-scatter grads + all-gather params ≈ 2Ψ + 2Ψ
  ZeRO-3 : all-gather params (fwd) + all-gather (bwd) + reduce-scatter grads
           ≈ 2Ψ + 2Ψ + 2Ψ
"""
from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORLD_SIZE_DEFAULT = 32
BYTES_PARAM_FP16 = 2
BYTES_GRAD_FP16 = 2
BYTES_OPTIM = 12  # fp32 master (4) + m (4) + v (4)


class ZeroStage(str, Enum):
    DP = "dp"          # ZeRO-0 / plain data parallel
    ZERO1 = "zero1"
    ZERO2 = "zero2"
    ZERO3 = "zero3"


# ---------------------------------------------------------------------------
# Memory / communication accounting (closed form — matches DeepSpeed docs)
# ---------------------------------------------------------------------------


@dataclass
class MemoryBreakdown:
    stage: str
    world_size: int
    num_params: int
    params_bytes: float
    grads_bytes: float
    optim_bytes: float
    total_per_gpu_bytes: float
    total_cluster_bytes: float
    comm_bytes_per_step: float

    def as_gb(self) -> Dict[str, float]:
        g = 1024**3
        return {
            "params_GiB": self.params_bytes / g,
            "grads_GiB": self.grads_bytes / g,
            "optim_GiB": self.optim_bytes / g,
            "per_gpu_GiB": self.total_per_gpu_bytes / g,
            "cluster_GiB": self.total_cluster_bytes / g,
            "comm_GiB_per_step": self.comm_bytes_per_step / g,
        }


def memory_for_stage(
    stage: ZeroStage,
    num_params: int,
    world_size: int = WORLD_SIZE_DEFAULT,
) -> MemoryBreakdown:
    """Per-GPU resident memory + estimated communication volume."""
    n = max(world_size, 1)
    psi = num_params

    if stage == ZeroStage.DP:
        params = BYTES_PARAM_FP16 * psi
        grads = BYTES_GRAD_FP16 * psi
        optim = BYTES_OPTIM * psi
        comm = 2 * BYTES_GRAD_FP16 * psi  # all-reduce ≈ 2Ψ (fp16)
    elif stage == ZeroStage.ZERO1:
        params = BYTES_PARAM_FP16 * psi
        grads = BYTES_GRAD_FP16 * psi
        optim = BYTES_OPTIM * psi / n
        # all-reduce grads + broadcast/all-gather updated shard of params
        comm = 2 * BYTES_GRAD_FP16 * psi + BYTES_PARAM_FP16 * psi
    elif stage == ZeroStage.ZERO2:
        params = BYTES_PARAM_FP16 * psi
        grads = BYTES_GRAD_FP16 * psi / n
        optim = BYTES_OPTIM * psi / n
        # reduce-scatter grads + all-gather params after update
        comm = BYTES_GRAD_FP16 * psi + BYTES_PARAM_FP16 * psi
    elif stage == ZeroStage.ZERO3:
        params = BYTES_PARAM_FP16 * psi / n
        grads = BYTES_GRAD_FP16 * psi / n
        optim = BYTES_OPTIM * psi / n
        # all-gather params fwd + all-gather bwd + reduce-scatter grads
        comm = 2 * BYTES_PARAM_FP16 * psi + BYTES_GRAD_FP16 * psi
    else:
        raise ValueError(stage)

    total = params + grads + optim
    return MemoryBreakdown(
        stage=stage.value,
        world_size=n,
        num_params=psi,
        params_bytes=params,
        grads_bytes=grads,
        optim_bytes=optim,
        total_per_gpu_bytes=total,
        total_cluster_bytes=total * n,
        comm_bytes_per_step=comm,
    )


def lecture_30b_table(world_size: int = 8) -> Dict[str, MemoryBreakdown]:
    """Reproduce the lecture ladder for a ~30B model (Ψ = 30e9)."""
    psi = 30_000_000_000
    return {s.value: memory_for_stage(s, psi, world_size) for s in ZeroStage}


# ---------------------------------------------------------------------------
# Virtual GPU + demo model
# ---------------------------------------------------------------------------


@dataclass
class VirtualGPU:
    rank: int
    world_size: int
    # measured during a step
    resident_bytes: float = 0.0
    flops: float = 0.0
    comm_bytes: float = 0.0
    step_ms: float = 0.0
    loss: float = 0.0


class DemoMLP(nn.Module):
    """Small MLP — enough params to shard visibly across 32 ranks."""

    def __init__(self, width: int = 256, depth: int = 4, vocab: int = 64):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(vocab, width)]
        for _ in range(depth - 1):
            layers += [nn.GELU(), nn.Linear(width, width)]
        layers += [nn.GELU(), nn.Linear(width, vocab)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def shard_range(numel: int, rank: int, world_size: int) -> Tuple[int, int]:
    """Contiguous shard [start, end) for this rank (last rank absorbs remainder)."""
    base = numel // world_size
    rem = numel % world_size
    # first `rem` ranks get base+1
    if rank < rem:
        start = rank * (base + 1)
        end = start + base + 1
    else:
        start = rem * (base + 1) + (rank - rem) * base
        end = start + base
    return start, end


# ---------------------------------------------------------------------------
# One training step under each ZeRO stage (threaded across ranks)
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    stage: str
    world_size: int
    num_params: int
    accounting: MemoryBreakdown
    per_rank: List[VirtualGPU]
    wall_ms: float
    mean_loss: float
    # "compute intensity": flops / (flops + equivalent_comm_flops)
    # we treat 1 byte of comm as costing COMM_FLOP_EQUIV flops for a simple ratio
    compute_fraction: float
    notes: str


COMM_FLOP_EQUIV = 50.0  # toy: moving a byte ≈ this many FLOPs of work (bandwidth tax)


def _flatten_params(model: nn.Module) -> torch.Tensor:
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()


def _assign_flat(model: nn.Module, flat: torch.Tensor) -> None:
    torch.nn.utils.vector_to_parameters(flat, model.parameters())


def run_step_on_rank(
    rank: int,
    world_size: int,
    stage: ZeroStage,
    width: int,
    depth: int,
    batch: int,
    vocab: int,
    seed: int,
    shared_grads: Optional[torch.Tensor],
    barrier_box: Dict,
) -> VirtualGPU:
    """
    Simulate one micro-batch on virtual GPU `rank`.

    Real compute: forward + backward on a local batch.
    Sharding: we only *materialize* the tensors that this stage keeps resident,
    and we count communication bytes for the collectives the stage would issue.
    """
    t0 = time.perf_counter()
    torch.manual_seed(seed + rank)

    model = DemoMLP(width=width, depth=depth, vocab=vocab)
    psi = model.num_params()
    # local micro-batch (data parallel: each rank sees different data)
    x = torch.randn(batch, vocab)
    y = torch.randint(0, vocab, (batch,))

    # ---- forward / backward (always local) ----
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    model.zero_grad(set_to_none=True)
    loss.backward()

    # FLOPs estimate: 2 matmuls ≈ 2 * 2 * M*N*K per Linear; crude but comparative
    flops = 0.0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            # fwd + bwd ≈ 6 * batch * in * out
            flops += 6.0 * batch * m.in_features * m.out_features

    # Collect local flat grad for "all-reduce / reduce-scatter"
    flat_grad = torch.nn.utils.parameters_to_vector(
        [p.grad.reshape(-1) for p in model.parameters()]
    )

    # Deposit into shared buffer (rank 0 allocates); others wait then average.
    # This models the collective without needing torch.distributed.
    with barrier_box["lock"]:
        if barrier_box["grads"] is None:
            barrier_box["grads"] = flat_grad.detach().clone()
            barrier_box["count"] = 1
        else:
            barrier_box["grads"] += flat_grad.detach()
            barrier_box["count"] += 1
        barrier_box["arrived"] += 1
        if barrier_box["arrived"] == world_size:
            barrier_box["grads"] /= world_size
            barrier_box["event"].set()
    barrier_box["event"].wait()
    avg_grad = barrier_box["grads"]

    # Apply a fake Adam step on the *shard this rank owns* under ZeRO-1/2/3
    start, end = shard_range(psi, rank, world_size)
    shard_n = max(end - start, 0)

    # Memory residency for this stage (what stays on-device after the step)
    mem = memory_for_stage(stage, psi, world_size)
    # For ZeRO-3, params are only fully present during compute; peak ≈ all-gathered
    # We report *steady-state resident* as in the lecture tables, and note peak separately.
    resident = mem.total_per_gpu_bytes
    if stage == ZeroStage.ZERO3:
        # peak during forward: full fp16 params temporarily + shard of rest
        peak = BYTES_PARAM_FP16 * psi + (BYTES_GRAD_FP16 + BYTES_OPTIM) * psi / world_size
        resident = mem.total_per_gpu_bytes  # report steady-state; peak in notes via accounting

    # Optimizer update on owned shard (simulates ZeRO-1/2/3 update ownership)
    flat_w = _flatten_params(model)
    if shard_n > 0:
        g = avg_grad[start:end]
        # toy Adam: w -= lr * g / (sqrt(g^2)+eps)  — just to touch the shard
        flat_w[start:end] -= 1e-3 * g / (g.abs().sqrt() + 1e-8)
    # In DP/ZeRO-1/2 every rank must end with the same full weights.
    # Model all-gather of updated shards via shared buffer.
    with barrier_box["lock"]:
        if barrier_box["weights"] is None:
            barrier_box["weights"] = flat_w.detach().clone()
            # mark only this shard as authoritative under ZeRO; under DP all same
            barrier_box["w_count"] = 1
        else:
            if stage in (ZeroStage.ZERO1, ZeroStage.ZERO2, ZeroStage.ZERO3):
                barrier_box["weights"][start:end] = flat_w[start:end]
            else:
                # DP: already identical after same avg grad; keep rank0
                pass
            barrier_box["w_count"] += 1
        barrier_box["w_arrived"] += 1
        if barrier_box["w_arrived"] == world_size:
            barrier_box["w_event"].set()
    barrier_box["w_event"].wait()

    gpu = VirtualGPU(
        rank=rank,
        world_size=world_size,
        resident_bytes=resident,
        flops=flops,
        comm_bytes=mem.comm_bytes_per_step,
        step_ms=(time.perf_counter() - t0) * 1000.0,
        loss=float(loss.detach()),
    )
    # stash peak for zero3 on rank 0 via attribute
    if stage == ZeroStage.ZERO3:
        gpu.resident_bytes = mem.total_per_gpu_bytes
        setattr(gpu, "peak_bytes", peak)
    return gpu


def simulate_stage(
    stage: ZeroStage,
    world_size: int = WORLD_SIZE_DEFAULT,
    width: int = 256,
    depth: int = 4,
    batch: int = 8,
    vocab: int = 64,
    seed: int = 0,
) -> StepResult:
    """Launch `world_size` virtual GPUs (threads) for one training step."""
    import threading

    probe = DemoMLP(width=width, depth=depth, vocab=vocab)
    psi = probe.num_params()
    accounting = memory_for_stage(stage, psi, world_size)

    barrier_box = {
        "lock": threading.Lock(),
        "grads": None,
        "count": 0,
        "arrived": 0,
        "event": threading.Event(),
        "weights": None,
        "w_count": 0,
        "w_arrived": 0,
        "w_event": threading.Event(),
    }

    t0 = time.perf_counter()
    per_rank: List[VirtualGPU] = []
    with ThreadPoolExecutor(max_workers=world_size) as pool:
        futs = [
            pool.submit(
                run_step_on_rank,
                r,
                world_size,
                stage,
                width,
                depth,
                batch,
                vocab,
                seed,
                None,
                barrier_box,
            )
            for r in range(world_size)
        ]
        for f in as_completed(futs):
            per_rank.append(f.result())
    wall_ms = (time.perf_counter() - t0) * 1000.0
    per_rank.sort(key=lambda g: g.rank)

    mean_loss = sum(g.loss for g in per_rank) / world_size
    total_flops = sum(g.flops for g in per_rank)
    # communication tax: one collective volume (not × world_size — fabric moves it once)
    comm_tax = accounting.comm_bytes_per_step * COMM_FLOP_EQUIV
    compute_fraction = total_flops / (total_flops + comm_tax + 1e-12)

    notes = {
        ZeroStage.DP: "Full params+grads+optim on every GPU. Cheapest comm, fattest memory.",
        ZeroStage.ZERO1: "Optimizer (12B/param) sharded. Still full model+grads. Saves ~12Ψ(1-1/N).",
        ZeroStage.ZERO2: "Optimizer + grads sharded. Full params. Reduce-scatter grads.",
        ZeroStage.ZERO3: "Everything sharded. All-gather params to compute. Highest comm, lowest memory.",
    }[stage]

    return StepResult(
        stage=stage.value,
        world_size=world_size,
        num_params=psi,
        accounting=accounting,
        per_rank=per_rank,
        wall_ms=wall_ms,
        mean_loss=mean_loss,
        compute_fraction=compute_fraction,
        notes=notes,
    )


def compare_all_stages(
    world_size: int = WORLD_SIZE_DEFAULT,
    width: int = 256,
    depth: int = 4,
    batch: int = 8,
    **kwargs,
) -> Dict[str, StepResult]:
    out = {}
    for stage in ZeroStage:
        out[stage.value] = simulate_stage(
            stage, world_size=world_size, width=width, depth=depth, batch=batch, **kwargs
        )
    return out


def scale_report(
    param_counts: List[int],
    world_sizes: List[int] = (8, 32, 64),
) -> List[Dict]:
    """Closed-form memory ladder like the lecture Excel sheet."""
    rows = []
    for psi in param_counts:
        for n in world_sizes:
            for stage in ZeroStage:
                m = memory_for_stage(stage, psi, n)
                gb = m.as_gb()
                rows.append(
                    {
                        "params_B": psi / 1e9,
                        "world_size": n,
                        "stage": stage.value,
                        "per_gpu_GiB": gb["per_gpu_GiB"],
                        "cluster_GiB": gb["cluster_GiB"],
                        "comm_GiB": gb["comm_GiB_per_step"],
                    }
                )
    return rows


def format_bytes(n: float) -> str:
    for unit, denom in (("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024), ("B", 1)):
        if abs(n) >= denom or unit == "B":
            return f"{n / denom:.3f} {unit}"
    return f"{n} B"
