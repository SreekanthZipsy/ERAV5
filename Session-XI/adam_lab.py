"""
Session XI — Optimizers & LR schedules lab helpers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Adam by hand
# ---------------------------------------------------------------------------


@dataclass
class AdamHyper:
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    bias_correction: bool = True


def adam_step_hand(
    w: float,
    g: float,
    m: float,
    v: float,
    t: int,
    hp: AdamHyper,
) -> Dict[str, float]:
    """One Adam update. Returns m, v, m_hat, v_hat, step, w_new."""
    m = hp.beta1 * m + (1.0 - hp.beta1) * g
    v = hp.beta2 * v + (1.0 - hp.beta2) * (g * g)
    if hp.bias_correction:
        m_hat = m / (1.0 - hp.beta1**t)
        v_hat = v / (1.0 - hp.beta2**t)
    else:
        m_hat = m
        v_hat = v
    step = hp.lr * m_hat / (math.sqrt(v_hat) + hp.eps)
    w_new = w - step
    return {
        "m": m,
        "v": v,
        "m_hat": m_hat,
        "v_hat": v_hat,
        "step": step,
        "w": w_new,
    }


def adam_five_grads_vs_pytorch(
    w0: float = 0.5,
    grads: Optional[List[float]] = None,
    hp: Optional[AdamHyper] = None,
) -> Dict:
    """
    Reproduce Adam on one scalar weight for five gradients; compare to torch.optim.Adam.
    """
    if grads is None:
        grads = [1.0, -0.5, 0.25, 2.0, -1.5]
    if hp is None:
        hp = AdamHyper()

    # --- hand ---
    hand_rows = []
    w, m, v = w0, 0.0, 0.0
    for t, g in enumerate(grads, start=1):
        out = adam_step_hand(w, g, m, v, t, hp)
        m, v, w = out["m"], out["v"], out["w"]
        hand_rows.append({"t": t, "g": g, **out})

    # --- PyTorch ---
    p = nn.Parameter(torch.tensor([w0], dtype=torch.float64))
    opt = torch.optim.Adam(
        [p], lr=hp.lr, betas=(hp.beta1, hp.beta2), eps=hp.eps, maximize=False
    )
    # torch Adam always uses bias correction; we'll compare with bias_correction=True
    torch_rows = []
    for t, g in enumerate(grads, start=1):
        opt.zero_grad(set_to_none=True)
        p.grad = torch.tensor([g], dtype=torch.float64)
        # peek state after step via manual mirror using same formula as torch
        opt.step()
        st = opt.state[p]
        m_t = float(st["exp_avg"])
        v_t = float(st["exp_avg_sq"])
        # reconstruct hats as torch does
        bc1 = 1.0 - hp.beta1**t
        bc2 = 1.0 - hp.beta2**t
        m_hat = m_t / bc1
        v_hat = v_t / bc2
        step = hp.lr * m_hat / (math.sqrt(v_hat) + hp.eps)
        torch_rows.append(
            {
                "t": t,
                "g": g,
                "m": m_t,
                "v": v_t,
                "m_hat": m_hat,
                "v_hat": v_hat,
                "step": step,
                "w": float(p.data),
            }
        )

    # compare hand (with bias corr) to torch
    hp_bc = AdamHyper(
        lr=hp.lr, beta1=hp.beta1, beta2=hp.beta2, eps=hp.eps, bias_correction=True
    )
    hand_bc = []
    w, m, v = w0, 0.0, 0.0
    for t, g in enumerate(grads, start=1):
        out = adam_step_hand(w, g, m, v, t, hp_bc)
        m, v, w = out["m"], out["v"], out["w"]
        hand_bc.append({"t": t, "g": g, **out})

    diffs = []
    for h, trow in zip(hand_bc, torch_rows):
        diffs.append(
            {
                "t": h["t"],
                "dm": abs(h["m"] - trow["m"]),
                "dv": abs(h["v"] - trow["v"]),
                "dm_hat": abs(h["m_hat"] - trow["m_hat"]),
                "dv_hat": abs(h["v_hat"] - trow["v_hat"]),
                "dstep": abs(h["step"] - trow["step"]),
                "dw": abs(h["w"] - trow["w"]),
            }
        )

    max_abs = 0.0
    for d in diffs:
        for k, val in d.items():
            if k == "t":
                continue
            max_abs = max(max_abs, val)

    return {
        "grads": grads,
        "w0": w0,
        "hand_with_bias_corr": hand_bc,
        "pytorch": torch_rows,
        "diffs": diffs,
        "max_abs_diff": max_abs,
    }


def compare_bias_correction_trajectory(
    w0: float = 0.5,
    grads: Optional[List[float]] = None,
    steps: int = 20,
    hp: Optional[AdamHyper] = None,
    horizon: int = 5000,
) -> Dict:
    """
    Plot window = first `steps` (assignment: 20).
    Horizon for 'stops mattering' can be longer: β2=0.999 bias dies slowly.
    Uses constant gradient for the horizon estimate so the signal is clean;
    the 20-step plot still uses the five cycling grads.
    """
    if grads is None:
        grads = [1.0, -0.5, 0.25, 2.0, -1.5]
    if hp is None:
        hp = AdamHyper(lr=1e-2)

    def run(bias: bool, n: int, grad_seq) -> List[Dict]:
        rows = []
        w, m, v = w0, 0.0, 0.0
        h = AdamHyper(
            lr=hp.lr, beta1=hp.beta1, beta2=hp.beta2, eps=hp.eps, bias_correction=bias
        )
        for t in range(1, n + 1):
            g = grad_seq[(t - 1) % len(grad_seq)]
            out = adam_step_hand(w, g, m, v, t, h)
            m, v, w = out["m"], out["v"], out["w"]
            rows.append({"t": t, "w": w, "step": out["step"], "m_hat": out["m_hat"]})
        return rows

    with_bc = run(True, steps, grads)
    without = run(False, steps, grads)
    step_diffs = [abs(a["step"] - b["step"]) for a, b in zip(with_bc, without)]

    # Long-horizon: constant g so relative gap is monotone in the β2 factor
    with_h = run(True, horizon, [1.0])
    without_h = run(False, horizon, [1.0])
    stop_step = None
    for t in range(horizon):
        a = with_h[t]["step"]
        d = abs(a - without_h[t]["step"])
        if d / max(abs(a), 1e-12) < 0.01:
            rest_ok = all(
                abs(with_h[k]["step"] - without_h[k]["step"])
                / max(abs(with_h[k]["step"]), 1e-12)
                < 0.01
                for k in range(t, horizon)
            )
            if rest_ok:
                stop_step = t + 1
                break

    # Theoretical scales: 1-β1^t≈1 around tens of steps; 1-β2^t≈1 around thousands
    t_beta1 = math.ceil(math.log(0.01) / math.log(hp.beta1)) if hp.beta1 < 1 else None
    t_beta2 = math.ceil(math.log(0.01) / math.log(hp.beta2)) if hp.beta2 < 1 else None

    return {
        "with_bias_correction": with_bc,
        "without_bias_correction": without,
        "step_abs_diff": step_diffs,
        "steps_until_diff_immaterial": stop_step,
        "horizon": horizon,
        "t_beta1_99pct": t_beta1,
        "t_beta2_99pct": t_beta2,
        "rel_diff_at_20": step_diffs[min(19, len(step_diffs) - 1)]
        / max(abs(with_bc[min(19, len(with_bc) - 1)]["step"]), 1e-12),
        "criterion": (
            "constant-g horizon: rel |Δstep|/|step_bc| < 1% for all remaining steps; "
            "first-20 plot uses the five cycling grads"
        ),
    }


# ---------------------------------------------------------------------------
# Tiny LM for schedule / sweep experiments
# ---------------------------------------------------------------------------


@dataclass
class LMConfig:
    vocab_size: int = 64
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 64


class Block(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
        )
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size))[None, None],
        )

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(y)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyLM(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.ln_f(x))


def ce_loss(logits, idx):
    V = logits.size(-1)
    return F.cross_entropy(logits[:, :-1].reshape(-1, V), idx[:, 1:].reshape(-1))


# ---------------------------------------------------------------------------
# LR schedules
# ---------------------------------------------------------------------------


def lr_cosine(step: int, total: int, warmup: int, lr_max: float, lr_min: float) -> float:
    if step < warmup:
        return lr_max * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


def lr_wsd(
    step: int,
    total: int,
    warmup: int,
    decay_start: int,
    lr_max: float,
    lr_min: float,
) -> float:
    """Warmup → Stable (constant) → Decay (linear to lr_min)."""
    if step < warmup:
        return lr_max * (step + 1) / max(warmup, 1)
    if step < decay_start:
        return lr_max
    # decay phase
    denom = max(total - decay_start, 1)
    progress = (step - decay_start) / denom
    progress = min(max(progress, 0.0), 1.0)
    return lr_max + (lr_min - lr_max) * progress


def set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for g in opt.param_groups:
        g["lr"] = lr


# ---------------------------------------------------------------------------
# Update-to-weight ratio + warmup
# ---------------------------------------------------------------------------


def layer_update_ratios(model: nn.Module, before: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """||ΔW|| / ||W|| per named parameter (leaf tensors)."""
    out = {}
    for name, p in model.named_parameters():
        if name not in before:
            continue
        delta = (p.data - before[name]).norm().item()
        base = before[name].norm().item()
        out[name] = delta / max(base, 1e-12)
    return out


def make_patterned_data(n: int, vocab: int, period: int = 8) -> torch.Tensor:
    """Simple repeating arithmetic pattern so CE can actually drop."""
    t = torch.arange(n)
    return (t % period + (t // period) % max(vocab - period, 1)) % vocab


def train_log_update_ratios(
    steps: int = 80,
    warmup: int = 20,
    lr_max: float = 3e-3,
    seed: int = 0,
) -> Dict:
    torch.manual_seed(seed)
    cfg = LMConfig(n_embd=64, n_layer=2, n_head=2, vocab_size=48, block_size=32)
    model = TinyLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr_max, betas=(0.9, 0.999))
    data = make_patterned_data(4000, cfg.vocab_size)

    hist = {
        "step": [],
        "lr": [],
        "loss": [],
        "ratio_mean": [],
        "ratio_median": [],
        "ratios": [],
    }
    for step in range(steps):
        # linear warmup then hold — isolates warmup from cosine decay
        if step < warmup:
            lr = lr_max * (step + 1) / warmup
        else:
            lr = lr_max
        set_lr(opt, lr)
        start = (step * 13) % (len(data) - 32)
        idx = data[start : start + 32].view(1, -1)
        before = {n: p.data.clone() for n, p in model.named_parameters()}
        opt.zero_grad(set_to_none=True)
        loss = ce_loss(model(idx), idx)
        loss.backward()
        opt.step()
        ratios = layer_update_ratios(model, before)
        vals = list(ratios.values())
        mean_r = sum(vals) / max(len(vals), 1)
        med_r = sorted(vals)[len(vals) // 2]
        hist["step"].append(step)
        hist["lr"].append(lr)
        hist["loss"].append(float(loss.detach()))
        hist["ratio_mean"].append(mean_r)
        hist["ratio_median"].append(med_r)
        hist["ratios"].append(ratios)

    # Warmup stops changing the ratio when LR stops rising (end of warmup).
    # Confirm: during warmup median ratio tracks rising LR; after warmup the
    # relative step-to-step change of median ratio drops below mid-warmup level.
    med = hist["ratio_median"]
    # skip step 0 (Adam first-step / init transient)
    d_warm = [abs(med[i] - med[i - 1]) for i in range(2, warmup)]
    warm_level = sorted(d_warm)[len(d_warm) // 2] if d_warm else 0.0
    warmup_effect_ends = warmup
    for i in range(warmup, len(med) - 1):
        # first post-warmup step where |Δmedian| is below typical warmup churn
        if abs(med[i] - med[i - 1]) <= warm_level:
            warmup_effect_ends = i
            break

    return {
        "hist": hist,
        "warmup_steps_scheduled": warmup,
        "step_warmup_stops_changing_ratio": warmup_effect_ends,
        "note": (
            "Per-layer ||ΔW||/||W|| logged each step. Warmup lifts LR → lifts update "
            "ratio; once LR plateaus (step==warmup), warmup no longer drives the ratio."
        ),
    }


# ---------------------------------------------------------------------------
# Cosine vs WSD
# ---------------------------------------------------------------------------


def train_schedule(
    schedule: str,
    total_steps: int = 300,
    stop_at: int = 200,
    lr_max: float = 3e-3,
    warmup: int = 30,
    seed: int = 0,
) -> Dict:
    torch.manual_seed(seed)
    cfg = LMConfig(n_embd=128, n_layer=2, n_head=4, vocab_size=64, block_size=64)
    model = TinyLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=0.01)
    data = make_patterned_data(8000, cfg.vocab_size, period=8)
    # WSD: stable until 80% of planned total, then decay
    decay_start = int(0.8 * total_steps)

    losses = []
    lrs = []
    for step in range(total_steps):
        if schedule == "cosine":
            lr = lr_cosine(step, total_steps, warmup, lr_max, lr_min=lr_max * 0.05)
        elif schedule == "wsd":
            lr = lr_wsd(step, total_steps, warmup, decay_start, lr_max, lr_min=lr_max * 0.05)
        else:
            raise ValueError(schedule)
        set_lr(opt, lr)
        start = (step * 17) % (len(data) - 64)
        idx = data[start : start + 64].view(1, -1)
        opt.zero_grad(set_to_none=True)
        loss = ce_loss(model(idx), idx)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        lrs.append(lr)

    return {
        "schedule": schedule,
        "losses": losses,
        "lrs": lrs,
        "loss_at_step_200": losses[stop_at - 1],
        "mean_loss_last10_at_200": sum(losses[stop_at - 10 : stop_at]) / 10,
        "loss_at_300": losses[-1],
        "mean_loss_last10_at_300": sum(losses[-10:]) / 10,
        "decay_start": decay_start if schedule == "wsd" else None,
        "warmup": warmup,
        "lr_max": lr_max,
    }


# ---------------------------------------------------------------------------
# LR sweep × width (μP-flavoured observation)
# ---------------------------------------------------------------------------


def lr_sweep_width(
    widths: List[int] = None,
    lrs: List[float] = None,
    steps: int = 80,
    seed: int = 0,
) -> Dict:
    if widths is None:
        widths = [256, 512, 1024]
    if lrs is None:
        # log-spaced — must tune each width (assignment warning)
        lrs = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]

    results = {w: [] for w in widths}
    for w in widths:
        for lr in lrs:
            torch.manual_seed(seed)
            n_head = 4 if w >= 256 else 2
            while w % n_head != 0:
                n_head -= 1
            cfg = LMConfig(
                vocab_size=64,
                n_layer=2,
                n_head=n_head,
                n_embd=w,
                block_size=64,
            )
            model = TinyLM(cfg)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
            data = make_patterned_data(6000, cfg.vocab_size, period=8)
            warmup = 10
            last_losses = []
            for step in range(steps):
                # short warmup then flat — fair compare of peak LR
                cur = lr * min(1.0, (step + 1) / warmup)
                set_lr(opt, cur)
                start = (step * 19) % (len(data) - 64)
                idx = data[start : start + 64].view(1, -1)
                opt.zero_grad(set_to_none=True)
                loss = ce_loss(model(idx), idx)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if step >= steps - 20:
                    last_losses.append(float(loss.detach()))
            mean_loss = sum(last_losses) / len(last_losses)
            results[w].append({"lr": lr, "loss": mean_loss})

    minima = {}
    for w, rows in results.items():
        best = min(rows, key=lambda r: r["loss"])
        minima[w] = best

    # Extrapolate to 4096: fit log(lr*) = a + b log(width)
    ws = sorted(minima.keys())
    log_w = [math.log(w) for w in ws]
    log_lr = [math.log(minima[w]["lr"]) for w in ws]
    n = len(ws)
    mean_x = sum(log_w) / n
    mean_y = sum(log_lr) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_w, log_lr))
    den = sum((x - mean_x) ** 2 for x in log_w) or 1.0
    b = num / den
    a = mean_y - b * mean_x
    lr_4096_fit = math.exp(a + b * math.log(4096))
    # Also report 1/width transfer from the best-measured width (common SP heuristic)
    ref_w = max(ws)
    lr_4096_inv = minima[ref_w]["lr"] * (ref_w / 4096)
    # μP-style transfer: keep η* from proxy if parametrization transfers (flat η*)
    lr_4096_mup = minima[ref_w]["lr"]

    # confidence from (a) interior minima not on grid edge, (b) slope consistency
    on_edge = any(
        minima[w]["lr"] in (min(lrs), max(lrs)) for w in ws
    )
    if on_edge:
        confidence = "low — at least one minimum sat on the LR grid edge; re-sweep finer"
    elif abs(b + 1.0) < 0.35:
        confidence = "medium — near 1/width; use inv-width transfer carefully"
    elif abs(b) < 0.35:
        confidence = "medium — near width-independent (μP-like); reuse proxy η*"
    else:
        confidence = "low — unstable scaling across only three widths"

    return {
        "grid": results,
        "minima": {str(w): minima[w] for w in minima},
        "fit_log_lr_vs_log_width": {"intercept": a, "slope": b},
        "lr_at_4096_extrapolated": lr_4096_fit,
        "lr_at_4096_inv_width_from_1024": lr_4096_inv,
        "lr_at_4096_mup_transfer": lr_4096_mup,
        "confidence": confidence,
        "note": (
            "Both sides tuned: same steps/warmup/seed/clip/data per (width, lr). "
            "Almost every optimizer claim that failed to replicate was a well-tuned "
            "method measured against a badly tuned one."
        ),
    }
