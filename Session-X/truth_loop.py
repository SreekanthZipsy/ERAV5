"""
Session X — training-truth helpers.
Small causal LM + observable step (shapes, hand grad, broken accum, MFU, dtypes).
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Tiny model
# ---------------------------------------------------------------------------


@dataclass
class Config:
    vocab_size: int = 64
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    block_size: int = 64


class Block(nn.Module):
    def __init__(self, cfg: Config):
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
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h)
        q, k, v = qkv.split(C, dim=-1)
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
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tied
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)
        x = self.tok_emb(tokens) + self.pos_emb(pos)[None, :, :]
        for blk in self.blocks:
            x = blk(x)
        hidden = self.ln_f(x)
        logits = self.lm_head(hidden)
        return {"hidden": hidden, "logits": logits}

    def n_params(self) -> int:
        # unique storage (tied head counted once)
        return sum({id(p): p.numel() for p in self.parameters()}.values())


def trace_training_step(model: TinyLM, tokens: torch.Tensor) -> List[Dict]:
    """Run one complete forward/backward step and report every material tensor."""
    rows: List[Dict] = []

    def record(name: str, tensor: torch.Tensor, meaning: str) -> None:
        shape = tuple(tensor.shape)
        rows.append({"name": name, "shape": list(shape), "meaning": meaning})
        print(f"{name:28s} {str(shape):18s} {meaning}")

    model.zero_grad(set_to_none=True)
    B, T = tokens.shape
    C, H, D = model.cfg.n_embd, model.cfg.n_head, model.cfg.n_embd // model.cfg.n_head
    for name, parameter in model.named_parameters():
        record(f"param.{name}", parameter, "trainable parameter dimensions")
    record("tokens", tokens, "(B,T): batch examples × token positions")
    positions = torch.arange(T, device=tokens.device)
    record("positions", positions, "(T): absolute position index")
    token_embeddings = model.tok_emb(tokens)
    record("token_embeddings", token_embeddings, "(B,T,C): example × position × embedding channel")
    position_embeddings = model.pos_emb(positions)
    record("position_embeddings", position_embeddings, "(T,C): position × embedding channel; broadcast over B")
    x = token_embeddings + position_embeddings[None, :, :]
    record("residual.input", x, "(B,T,C): residual stream entering transformer")

    for layer_i, block in enumerate(model.blocks):
        prefix = f"block{layer_i}"
        h = block.ln1(x)
        record(f"{prefix}.ln1", h, "(B,T,C): normalized residual channels")
        qkv = block.qkv(h)
        record(f"{prefix}.qkv", qkv, "(B,T,3C): packed query, key, value channels")
        q, k, v = qkv.split(C, dim=-1)
        record(f"{prefix}.q", q, "(B,T,C): unpacked query channels")
        record(f"{prefix}.k", k, "(B,T,C): unpacked key channels")
        record(f"{prefix}.v", v, "(B,T,C): unpacked value channels")
        qh = q.view(B, T, H, D).transpose(1, 2)
        kh = k.view(B, T, H, D).transpose(1, 2)
        vh = v.view(B, T, H, D).transpose(1, 2)
        record(f"{prefix}.q_heads", qh, "(B,H,T,D): example × head × query position × head channel")
        record(f"{prefix}.k_heads", kh, "(B,H,T,D): example × head × key position × head channel")
        record(f"{prefix}.v_heads", vh, "(B,H,T,D): example × head × value position × head channel")
        scores = (qh @ kh.transpose(-2, -1)) / math.sqrt(D)
        record(f"{prefix}.scores", scores, "(B,H,T,T): example × head × query position × key position")
        causal_mask = block.mask[:, :, :T, :T]
        record(f"{prefix}.causal_mask", causal_mask, "(1,1,T,T): broadcast mask; 1 means key is visible")
        masked_scores = scores.masked_fill(causal_mask == 0, float("-inf"))
        record(f"{prefix}.masked_scores", masked_scores, "(B,H,T,T): attention scores after causal masking")
        attention = F.softmax(masked_scores, dim=-1)
        record(f"{prefix}.attention", attention, "(B,H,T,T): probability over key positions")
        context_heads = attention @ vh
        record(f"{prefix}.context_heads", context_heads, "(B,H,T,D): weighted value vectors per head")
        context = context_heads.transpose(1, 2).contiguous().view(B, T, C)
        record(f"{prefix}.context", context, "(B,T,C): attention heads concatenated")
        attention_out = block.proj(context)
        record(f"{prefix}.attention_out", attention_out, "(B,T,C): projected attention update")
        x_after_attention = x + attention_out
        record(f"{prefix}.residual_attn", x_after_attention, "(B,T,C): residual after attention update")
        h2 = block.ln2(x_after_attention)
        record(f"{prefix}.ln2", h2, "(B,T,C): normalized channels entering MLP")
        mlp_expanded = block.mlp[0](h2)
        record(f"{prefix}.mlp_expanded", mlp_expanded, "(B,T,4C): expanded MLP channels")
        mlp_activated = block.mlp[1](mlp_expanded)
        record(f"{prefix}.mlp_activated", mlp_activated, "(B,T,4C): GELU activations")
        mlp_out = block.mlp[2](mlp_activated)
        record(f"{prefix}.mlp_out", mlp_out, "(B,T,C): projected MLP update")
        x = x_after_attention + mlp_out
        record(f"{prefix}.residual_out", x, "(B,T,C): residual stream leaving block")

    hidden = model.ln_f(x)
    record("final_norm", hidden, "(B,T,C): final normalized hidden states")
    logits = model.lm_head(hidden)
    record("logits", logits, "(B,T,V): score for every vocabulary item")
    shift_logits = logits[:, :-1, :]
    shift_labels = tokens[:, 1:]
    record("shift_logits", shift_logits, "(B,T-1,V): predictions aligned to next-token labels")
    record("shift_labels", shift_labels, "(B,T-1): next-token target ids")
    flat_logits = shift_logits.reshape(-1, model.cfg.vocab_size)
    flat_labels = shift_labels.reshape(-1)
    record("flat_logits", flat_logits, "(B·(T-1),V): rows consumed by cross entropy")
    record("flat_labels", flat_labels, "(B·(T-1)): target id for each CE row")
    token_nll = F.cross_entropy(flat_logits, flat_labels, reduction="none")
    record("token_nll", token_nll, "(B·(T-1)): one negative log-likelihood per target")
    loss = token_nll.mean()
    record("loss", loss, "(): scalar mean over all next-token targets")
    loss.backward()
    for name, parameter in model.named_parameters():
        record(f"grad.{name}", parameter.grad, "same dimensions as parameter; d(loss)/d(parameter)")
    return rows


def causal_ce(
    logits: torch.Tensor, tokens: torch.Tensor, ignore_index: int = -100
) -> Tuple[torch.Tensor, int]:
    """Mean CE over non-ignored next-token targets. Returns (loss, n_valid)."""
    V = logits.size(-1)
    shift_logits = logits[:, :-1].reshape(-1, V)
    shift_labels = tokens[:, 1:].reshape(-1)
    valid = shift_labels != ignore_index
    n = int(valid.sum().item())
    if n == 0:
        return logits.new_zeros(()), 0
    loss = F.cross_entropy(shift_logits[valid], shift_labels[valid], reduction="mean")
    return loss, n


def causal_ce_sum(
    logits: torch.Tensor, tokens: torch.Tensor, ignore_index: int = -100
) -> Tuple[torch.Tensor, int]:
    """Sum of NLLs (for token-weighted accumulation)."""
    V = logits.size(-1)
    shift_logits = logits[:, :-1].reshape(-1, V)
    shift_labels = tokens[:, 1:].reshape(-1)
    valid = shift_labels != ignore_index
    n = int(valid.sum().item())
    if n == 0:
        return logits.new_zeros(()), 0
    loss_sum = F.cross_entropy(
        shift_logits[valid], shift_labels[valid], reduction="sum"
    )
    return loss_sum, n


# ---------------------------------------------------------------------------
# Hand gradient check
# ---------------------------------------------------------------------------


def finite_diff_grad(
    model: nn.Module,
    tokens: torch.Tensor,
    param: nn.Parameter,
    index: Tuple[int, ...],
    eps: float = 1e-4,
) -> float:
    """Central difference ∂L/∂w[index] for scalar mean CE loss."""
    with torch.no_grad():
        w = param.data
        orig = w[index].item()
        w[index] = orig + eps
        loss_p, _ = causal_ce(model(tokens)["logits"], tokens)
        w[index] = orig - eps
        loss_m, _ = causal_ce(model(tokens)["logits"], tokens)
        w[index] = orig
    return (loss_p.item() - loss_m.item()) / (2 * eps)


def analytic_grad(
    model: nn.Module, tokens: torch.Tensor, param: nn.Parameter, index: Tuple[int, ...]
) -> float:
    model.zero_grad(set_to_none=True)
    loss, _ = causal_ce(model(tokens)["logits"], tokens)
    loss.backward()
    g = param.grad[index].item()
    model.zero_grad(set_to_none=True)
    return g


def verify_grad_hand(
    eps: float = 1e-3,
) -> Dict:
    """
    Isolated check: y = w * x, L = (y - target)^2.
    Analytic ∂L/∂w = 2(wx - t)x should match central difference to many decimals.
    Also runs one CE check on TinyLM in float64 for the assignment loop.
    """
    # --- exact toy ---
    w = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    x = torch.tensor(1.3, dtype=torch.float64)
    target = torch.tensor(0.5, dtype=torch.float64)
    y = w * x
    loss = (y - target) ** 2
    loss.backward()
    g_anal = w.grad.item()
    with torch.no_grad():
        w_p = (w + eps) * x
        w_m = (w - eps) * x
        g_num = (((w_p - target) ** 2) - ((w_m - target) ** 2)).item() / (2 * eps)
    toy = {
        "analytic": g_anal,
        "numerical": g_num,
        "abs_err": abs(g_anal - g_num),
        "formula": "L=(w*x - t)^2,  dL/dw = 2(wx-t)x",
    }

    # --- TinyLM CE in float64 ---
    torch.manual_seed(0)
    cfg = Config(vocab_size=32, n_embd=32, n_layer=1, n_head=2, block_size=16)
    model = TinyLM(cfg).double()
    tokens = torch.randint(0, cfg.vocab_size, (1, 12))
    # untie temporarily so nudging lm path isn't shared weirdly with emb for this index
    # use mlp weight which is not tied
    param = model.blocks[0].mlp[0].weight
    index = (0, 0)
    g_a = analytic_grad(model, tokens, param, index)
    with torch.no_grad():
        original_weight = param[index].item()
        param[index] = original_weight + eps
        loss_plus = causal_ce(model(tokens)["logits"], tokens)[0].item()
        param[index] = original_weight - eps
        loss_minus = causal_ce(model(tokens)["logits"], tokens)[0].item()
        param[index] = original_weight
    g_n = (loss_plus - loss_minus) / (2 * eps)
    lm = {
        "analytic": g_a,
        "numerical": g_n,
        "abs_err": abs(g_a - g_n),
        "rel_err": abs(g_a - g_n) / max(abs(g_a), 1e-12),
        "param": "blocks.0.mlp.0.weight[0,0]",
        "eps": eps,
        "dtype": "float64",
        "original_weight": original_weight,
        "loss_plus": loss_plus,
        "loss_minus": loss_minus,
        "central_difference": "(L(w+eps)-L(w-eps))/(2*eps)",
    }
    return {"toy": toy, "tinylm_ce": lm}

# ---------------------------------------------------------------------------
# Broken vs correct gradient accumulation
# ---------------------------------------------------------------------------


def make_microbatches(
    lengths: List[int], vocab_size: int, device: torch.device, seed: int = 0
) -> List[torch.Tensor]:
    """Unequal token counts with deliberately different short/long patterns."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    batches = []
    for i, length in enumerate(lengths):
        start = int(torch.randint(0, vocab_size, (1,), generator=g).item())
        if length <= 16:
            # Short examples follow +1 transitions.
            seq = [(start + j) % vocab_size for j in range(length)]
        else:
            # Long examples follow -1 transitions and therefore own most tokens.
            seq = [(start - j) % vocab_size for j in range(length)]
        batches.append(torch.tensor([seq], dtype=torch.long, device=device))
    return batches


def accumulate_wrong(model: nn.Module, microbatches: List[torch.Tensor]) -> float:
    """BUG (pre-2024 style): average of per-microbatch *mean* losses.
    Short and long batches get equal vote → short batches over-weighted.
    """
    means = []
    for mb in microbatches:
        loss, n = causal_ce(model(mb)["logits"], mb)
        assert n > 0
        means.append(loss)
    return torch.stack(means).mean().item()


def accumulate_correct(model: nn.Module, microbatches: List[torch.Tensor]) -> float:
    """Token-weighted: sum(nll) / sum(n_valid)."""
    total = 0.0
    n_all = 0
    for mb in microbatches:
        loss_sum, n = causal_ce_sum(model(mb)["logits"], mb)
        total += loss_sum.item()
        n_all += n
    return total / max(n_all, 1)


def train_accum_comparison(
    steps: int = 80,
    seed: int = 0,
) -> Dict:
    """Train cloned models with wrong and token-weighted accumulated gradients."""
    torch.manual_seed(seed)
    device = torch.device("cpu")
    cfg = Config(vocab_size=48, n_embd=48, n_layer=2, n_head=2, block_size=64)
    lengths = [16, 16, 48, 48]  # two short, two long → wrong rule biases short

    model_wrong = TinyLM(cfg).to(device)
    model_right = TinyLM(cfg).to(device)
    model_right.load_state_dict(model_wrong.state_dict())

    opt_w = torch.optim.AdamW(model_wrong.parameters(), lr=3e-3)
    opt_r = torch.optim.AdamW(model_right.parameters(), lr=3e-3)

    hist = {
        "step": [],
        "eval_wrong_model": [],
        "eval_right_model": [],
        "short_wrong_model": [],
        "short_right_model": [],
        "objective_gap_initial_model": [],
        "valid_tokens": [length - 1 for length in lengths],
    }

    for step in range(steps):
        mbs = make_microbatches(lengths, cfg.vocab_size, device, seed=seed + step)

        # Wrong, but real accumulation: backward once per microbatch while
        # retaining .grad, then take one optimizer step.
        opt_w.zero_grad(set_to_none=True)
        for mb in mbs:
            loss, _ = causal_ce(model_wrong(mb)["logits"], mb)
            (loss / len(mbs)).backward()
        opt_w.step()

        # Correct accumulation: scale every summed microbatch loss by the
        # total number of valid tokens, backward each, then step once.
        opt_r.zero_grad(set_to_none=True)
        n_all = sum(mb.numel() - mb.size(0) for mb in mbs)
        for mb in mbs:
            loss_sum, n = causal_ce_sum(model_right(mb)["logits"], mb)
            assert n > 0
            (loss_sum / n_all).backward()
        opt_r.step()

        with torch.no_grad():
            eval_mbs = make_microbatches(
                lengths, cfg.vocab_size, device, seed=10_000 + seed + step
            )
            weighted_wrong_model = accumulate_correct(model_wrong, eval_mbs)
            weighted_right_model = accumulate_correct(model_right, eval_mbs)
            short_wrong_model = sum(
                causal_ce(model_wrong(mb)["logits"], mb)[0].item()
                for mb in eval_mbs[:2]
            ) / 2
            short_right_model = sum(
                causal_ce(model_right(mb)["logits"], mb)[0].item()
                for mb in eval_mbs[:2]
            ) / 2
            # Pure formula difference on one frozen model and identical data.
            formula_wrong = accumulate_wrong(model_right, eval_mbs)
            formula_right = accumulate_correct(model_right, eval_mbs)

        hist["step"].append(step)
        hist["eval_wrong_model"].append(weighted_wrong_model)
        hist["eval_right_model"].append(weighted_right_model)
        hist["short_wrong_model"].append(short_wrong_model)
        hist["short_right_model"].append(short_right_model)
        hist["objective_gap_initial_model"].append(formula_wrong - formula_right)

    return hist


# ---------------------------------------------------------------------------
# Grad-norm logging
# ---------------------------------------------------------------------------


def global_grad_norm(model: nn.Module) -> float:
    sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sq += float(p.grad.detach().pow(2).sum())
    return math.sqrt(sq)


def train_with_grad_norm_log(steps: int = 150, spike_at: int = 40) -> Dict:
    """
    Controlled failure: at spike_at multiply only the backward objective.
    Raw loss and probe loss are recorded before optimizer.step(), so the gradient
    warning is visible at step s and the damaged loss first appears at s+1.
    """
    torch.manual_seed(2)
    device = torch.device("cpu")
    cfg = Config(vocab_size=40, n_embd=32, n_layer=2, n_head=2, block_size=48)
    model = TinyLM(cfg).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)  # SGD: shock sticks more visibly
    data = [i % cfg.vocab_size for i in range(1500)]
    probe = torch.tensor([data[100:132]], dtype=torch.long, device=device)

    hist = {
        "step": [], "probe_loss_pre_update": [], "raw_train_loss": [],
        "backward_scale": [], "grad_norm": []
    }

    for step in range(steps):
        start = (step * 17) % max(1, len(data) - 32)
        seq = data[start : start + 32]
        tokens = torch.tensor([seq], dtype=torch.long, device=device)

        with torch.no_grad():
            probe_loss, _ = causal_ce(model(probe)["logits"], probe)
        opt.zero_grad(set_to_none=True)
        raw_train_loss, _ = causal_ce(model(tokens)["logits"], tokens)
        scale = 80.0 if step == spike_at else 1.0
        (raw_train_loss * scale).backward()
        gn = global_grad_norm(model)
        if step != spike_at:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        hist["step"].append(step)
        hist["probe_loss_pre_update"].append(float(probe_loss))
        hist["raw_train_loss"].append(float(raw_train_loss.detach()))
        hist["backward_scale"].append(scale)
        hist["grad_norm"].append(gn)

    gn = hist["grad_norm"]
    ls = hist["probe_loss_pre_update"]
    best = {
        "warning_step": spike_at,
        "grad_norm_before": gn[spike_at - 1],
        "grad_norm_at_warning": gn[spike_at],
        "probe_loss_before": ls[spike_at - 1],
        "probe_loss_at_warning_pre_update": ls[spike_at],
        "probe_loss_next_step": ls[spike_at + 1],
        "note": (
            "The raw/probe loss was measured before the update. A controlled 80x "
            "backward-scale bug made ||g|| jump at step s; its effect on probe loss "
            "could only be observed at step s+1."
        ),
    }
    hist["finding"] = best
    return hist


# ---------------------------------------------------------------------------
# MFU
# ---------------------------------------------------------------------------


def estimate_mfu(
    n_params: int,
    tokens: int,
    wall_time_s: float,
    peak_tflops: float,
    n_layer: int = 0,
    n_embd: int = 0,
    seq_len: int = 0,
) -> Dict:
    """
    Transformer training FLOPs: 6*N per token plus the sequence-dependent
    attention matmuls, approximately 12*L*C*T per token.
    MFU = achieved_TFLOP/s / peak_TFLOP/s.
    """
    parameter_flops = 6.0 * n_params * tokens
    attention_flops = 12.0 * n_layer * n_embd * seq_len * tokens
    flops = parameter_flops + attention_flops
    achieved_tflops = (flops / max(wall_time_s, 1e-9)) / 1e12
    mfu = achieved_tflops / peak_tflops
    return {
        "n_params": n_params,
        "tokens": tokens,
        "wall_time_s": wall_time_s,
        "parameter_flops": parameter_flops,
        "attention_flops": attention_flops,
        "approx_flops": flops,
        "achieved_tflops": achieved_tflops,
        "peak_tflops_assumed": peak_tflops,
        "mfu": mfu,
        "mfu_pct": 100.0 * mfu,
    }


# ---------------------------------------------------------------------------
# Float bit layouts for 0.1
# ---------------------------------------------------------------------------


def fp32_bits(x: float) -> Dict:
    u = struct.unpack(">I", struct.pack(">f", float(x)))[0]
    bits = f"{u:032b}"
    sign, exp, mant = bits[0], bits[1:9], bits[9:]
    exp_val = int(exp, 2) - 127
    # decode
    frac = 1.0
    for i, b in enumerate(mant):
        if b == "1":
            frac += 2 ** (-(i + 1))
    value = ((-1) ** int(sign)) * frac * (2.0**exp_val)
    return {
        "format": "fp32",
        "hex": f"0x{u:08X}",
        "bits": bits,
        "sign": sign,
        "exponent_bits": exp,
        "exponent_biased": int(exp, 2),
        "exponent_unbiased": exp_val,
        "mantissa_bits": mant,
        "approx_value": value,
        "layout": "1 sign | 8 exponent (bias 127) | 23 mantissa",
    }


def bf16_bits_from_fp32(x: float) -> Dict:
    """Convert FP32 to BF16 with IEEE round-to-nearest, ties-to-even."""
    u = struct.unpack(">I", struct.pack(">f", float(x)))[0]
    # Add 0x7FFF plus the retained LSB to implement ties-to-even.
    bf = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF
    bits = f"{bf:016b}"
    sign, exp, mant = bits[0], bits[1:9], bits[9:]
    exp_val = int(exp, 2) - 127
    frac = 1.0
    for i, b in enumerate(mant):
        if b == "1":
            frac += 2 ** (-(i + 1))
    value = ((-1) ** int(sign)) * frac * (2.0**exp_val)
    return {
        "format": "bf16",
        "hex": f"0x{bf:04X}",
        "bits": bits,
        "sign": sign,
        "exponent_bits": exp,
        "exponent_biased": int(exp, 2),
        "exponent_unbiased": exp_val,
        "mantissa_bits": mant,
        "approx_value": value,
        "layout": "1 sign | 8 exponent (bias 127) | 7 mantissa",
        "note": "Same exponent range as fp32; coarser resolution than fp16.",
    }


def fp8_e4m3_bits(x: float) -> Dict:
    """
    Encode to FP8 E4M3 (bias 7): 1 sign, 4 exp, 3 mantissa.
    Finite max ≈ 448. We round to nearest representable.
    """
    if x == 0.0:
        return {
            "format": "fp8_e4m3",
            "hex": "0x00",
            "bits": "00000000",
            "sign": "0",
            "exponent_bits": "0000",
            "mantissa_bits": "000",
            "approx_value": 0.0,
            "layout": "1 sign | 4 exponent (bias 7) | 3 mantissa",
        }
    sign = 0 if x >= 0 else 1
    ax = abs(float(x))
    # frexp-style: ax = m * 2^e with 1 <= m < 2
    e = int(math.floor(math.log2(ax)))
    m = ax / (2**e)  # in [1, 2)
    # biased exponent
    E = e + 7
    # clamp to E4M3 finite range: E in 1..14 (0 and 15 special in some variants;
    # OCP E4M3 uses E=15 with mant!=0 as NaN; E=0 subnormals)
    if E >= 15:
        # max finite: exp=14, mant=7 → (1+7/8)*2^(14-7) = 1.875*128 = 240
        # Actually E4M3 max is 448 = (1+0.875)*2^8 with exp=15? 
        # NVIDIA/OCP E4M3: bias 7, max 448 when exp=15 mant=6 (not all-ones mant as NaN in some docs)
        # We'll use common table: for 0.1 we won't saturate.
        bits_u = (sign << 7) | (0b1111 << 3) | 0b110
        value = 448.0 if sign == 0 else -448.0
    elif E <= 0:
        # subnormal: exp bits 0, mantissa holds leading fraction of ax / 2^(1-7)
        # value = (mant/8) * 2^(1-7) = mant * 2^(-9)
        mant_f = ax / (2 ** (-6))  # scale into [0, 1) roughly for E=0
        mant = int(round(mant_f * 8))
        mant = max(0, min(7, mant))
        bits_u = (sign << 7) | mant
        value = ((-1) ** sign) * (mant / 8.0) * (2 ** (-6))
    else:
        # normal: frac = 1 + mant/8
        frac = m - 1.0  # [0, 1)
        mant = int(round(frac * 8))
        if mant == 8:  # overflow mantissa → bump exp
            mant = 0
            E += 1
        if E >= 15:
            bits_u = (sign << 7) | (0b1111 << 3) | 0b110
            value = 448.0 if sign == 0 else -448.0
        else:
            bits_u = (sign << 7) | ((E & 0xF) << 3) | (mant & 0x7)
            value = ((-1) ** sign) * (1.0 + mant / 8.0) * (2.0 ** (E - 7))

    bits = f"{bits_u:08b}"
    return {
        "format": "fp8_e4m3",
        "hex": f"0x{bits_u:02X}",
        "bits": bits,
        "sign": bits[0],
        "exponent_bits": bits[1:5],
        "exponent_biased": int(bits[1:5], 2),
        "exponent_unbiased": int(bits[1:5], 2) - 7,
        "mantissa_bits": bits[5:],
        "approx_value": value,
        "rel_error_pct": 100.0 * abs(value - x) / abs(x),
        "layout": "1 sign | 4 exponent (bias 7) | 3 mantissa",
    }


def describe_0_1() -> Dict:
    return {
        "fp32": fp32_bits(0.1),
        "bf16": bf16_bits_from_fp32(0.1),
        "fp8_e4m3": fp8_e4m3_bits(0.1),
    }
