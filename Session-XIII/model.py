"""~21M-parameter GPT with a standard residual baseline and reversible variants.

Every variant shares the same block f(p) = Attn(LN1 p) + MLP(LN2(p + Attn(LN1 p))),
so parameter counts are identical; only the update rule across depth changes.

  baseline    p[l+1] = p[l] + f(p[l])                                  (not reversible)
  midpoint    p[l+1] = p[l-1] + 2h f(p[l])                             (paper eq. 4)
  midpoint_a  p[l+1] = a p[l-1] + (1-a) p[l] + h f(p[l]),  a = +-1 + U(-.5,.5)   (eq. 15)
  leapfrog    p[l+1] = 2 p[l] - p[l-1] + h^2 f(p[l])                   (eq. 6)
  euler       q[l+1] = q[l] + Attn(LN1 p[l]);  p[l+1] = p[l] + MLP(LN2 q[l+1])
              (Hamiltonian / symplectic-Euler coupling, eqs. 8-9 with a=b=1)

With rev_backward=True the reversible variants run under RevStack: the forward
keeps only the last two states and the backward reconstructs each layer's input
from its output, so activation memory no longer grows with depth.

With exact=True (default) midpoint, leapfrog and euler keep their states in int64
fixed point, which makes the reconstruction bit-exact under fp16 autocast.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

MODES = ("baseline", "midpoint", "midpoint_a", "leapfrog", "euler")
FRAC_BITS = 32
FIX_SCALE = float(2**FRAC_BITS)

# Fused fixed-point helpers. The forward (_fix_comb) and inverse (_fix_uncomb) must
# quantize c*y with the identical expression so the integer states cancel exactly.


@torch.compile(dynamic=False)
def _quant(x):
    return torch.round(x.float() * FIX_SCALE).to(torch.int64)


@torch.compile(dynamic=False)
def _dequant(s):
    return s.to(torch.float32) * (1.0 / FIX_SCALE)


@torch.compile(dynamic=False)
def _fix_comb(a, b, y, al: int, be: int, c: float):
    """al*a + be*b + Q(c*y), with a, b int64 fixed-point and al, be in {-1, 0, 1, 2}."""
    return al * a + be * b + torch.round(y.float() * c * FIX_SCALE).to(torch.int64)


@torch.compile(dynamic=False)
def _fix_uncomb(new, b, y, al: int, be: int, c: float):
    """Inverse of _fix_comb for the 'a' argument (al = +-1, so dividing equals multiplying)."""
    return (new - be * b - torch.round(y.float() * c * FIX_SCALE).to(torch.int64)) * al


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    block_size: int = 1024
    n_layer: int = 10
    n_head: int = 6
    n_embd: int = 384
    mode: str = "baseline"
    h: float = 0.5
    rev_backward: bool = True
    exact: bool = True
    ce_chunks: int = 1


class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.proj.is_residual_out = True

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.proj.is_residual_out = True

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def attn_branch(self, p):
        return self.attn(self.ln1(p))

    def mlp_branch(self, q):
        return self.mlp(self.ln2(q))

    def forward(self, p):
        a = self.attn_branch(p)
        return a + self.mlp_branch(p + a)


class RevStack(torch.autograd.Function):
    """Runs all blocks without storing activations; rebuilds them in backward."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, model):
        ex = model.exact
        with torch.no_grad():
            a, b = model.run_states(x, ex)
        ctx.model, ctx.ex = model, ex
        ctx.x_ref = x.detach().clone() if model.track_recon else None
        ctx.save_for_backward(a, b)
        return model.dec(b, ex)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, g_out):
        model, ex = ctx.model, ctx.ex
        a, b = ctx.saved_tensors
        gb = g_out.float()
        ga = torch.zeros_like(gb)
        for i in reversed(range(len(model.blocks))):
            a, b, ga, gb = model.step_back(i, a, b, ga, gb, ex)
        if ctx.x_ref is not None:
            ref = ctx.x_ref
            err = torch.maximum((model.dec(a, ex) - ref).abs().max(), (model.dec(b, ex) - ref).abs().max())
            model.last_recon_err = (err / ref.abs().max()).item()
        return ga + gb, None


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.mode in MODES
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        g = torch.Generator().manual_seed(1234)
        a = torch.where(torch.rand(cfg.n_layer, generator=g) < 0.5, 1.0, -1.0)
        a = a + (torch.rand(cfg.n_layer, generator=g) - 0.5)
        self.register_buffer("a_coef", a, persistent=True)
        self._a_list = a.tolist()
        self.track_recon = False
        self.last_recon_err = None
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            std = 0.02 / math.sqrt(2 * self.cfg.n_layer) if getattr(m, "is_residual_out", False) else 0.02
            nn.init.normal_(m.weight, 0.0, std)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wte.weight.numel() + self.wpe.weight.numel()
        return n

    # --- update rules: state is (a, b) = (p[l-1], p[l]) or (q, p) for euler ---
    #
    # With ex=True the states are int64 fixed-point numbers (FRAC_BITS fractional bits)
    # and every layer increment is quantized before being added. Integer add/subtract
    # is exact and the block recompute is bit-identical, so the backward pass rebuilds
    # the forward states exactly. midpoint_a multiplies by a non-integer a and cannot
    # be inverted exactly in integers, so it always uses float states.

    @property
    def exact(self):
        return self.cfg.exact and self.cfg.mode != "midpoint_a"

    @staticmethod
    def enc(x, ex):
        return _quant(x) if ex else x.float()

    @staticmethod
    def dec(s, ex):
        return _dequant(s) if ex else s

    def _coeffs(self, i, ex):
        m, h = self.cfg.mode, self.cfg.h
        if m == "midpoint":
            return (1, 0, 2 * h) if ex else (1.0, 0.0, 2 * h)
        if m == "leapfrog":
            return (-1, 2, h * h) if ex else (-1.0, 2.0, h * h)
        if m == "midpoint_a":
            al = self._a_list[i]
            return al, 1.0 - al, h
        raise ValueError(m)

    def step(self, i, a, b, ex):
        blk = self.blocks[i]
        if self.cfg.mode == "euler":
            if ex:
                q = _fix_comb(a, a, blk.attn_branch(_dequant(b)), 1, 0, 1.0)
                return q, _fix_comb(b, b, blk.mlp_branch(_dequant(q)), 1, 0, 1.0)
            q = a + blk.attn_branch(b).float()
            return q, b + blk.mlp_branch(q).float()
        al, be, c = self._coeffs(i, ex)
        y = blk(self.dec(b, ex))
        if ex:
            return b, _fix_comb(a, b, y, al, be, c)
        return b, al * a + be * b + c * y.float()

    def step_back(self, i, a, b, ga, gb, ex):
        """Given state after block i and its grads, return state before block i and its grads."""
        blk = self.blocks[i]
        if self.cfg.mode == "euler":
            q_new, p_new, gq, gp = a, b, ga, gb
            with torch.enable_grad():
                qq = self.dec(q_new, ex).detach().requires_grad_()
                m = blk.mlp_branch(qq)
                torch.autograd.backward(m, gp.to(m.dtype))
            p_old = _fix_uncomb(p_new, p_new, m.detach(), 1, 0, 1.0) if ex else p_new - m.detach().float()
            gq = gq + qq.grad.float()
            with torch.enable_grad():
                pp = self.dec(p_old, ex).detach().requires_grad_()
                at = blk.attn_branch(pp)
                torch.autograd.backward(at, gq.to(at.dtype))
            q_old = _fix_uncomb(q_new, q_new, at.detach(), 1, 0, 1.0) if ex else q_new - at.detach().float()
            gp = gp + pp.grad.float()
            return q_old, p_old, gq, gp
        al, be, c = self._coeffs(i, ex)
        with torch.enable_grad():
            pl = self.dec(a, ex).detach().requires_grad_()
            y = blk(pl)
            torch.autograd.backward(y, (c * gb).to(y.dtype))
        if ex:
            prev = _fix_uncomb(b, a, y.detach(), al, be, c)
        else:
            prev = (b - be * a - c * y.detach().float()) / al
        return prev, a, al * gb, ga + be * gb + pl.grad.float()

    def run_states(self, x, ex):
        a = b = self.enc(x, ex)
        for i in range(len(self.blocks)):
            a, b = self.step(i, a, b, ex)
        return a, b

    def body(self, x):
        if self.cfg.mode == "baseline":
            for blk in self.blocks:
                x = x + blk(x)
            return x
        if self.cfg.rev_backward and torch.is_grad_enabled():
            return RevStack.apply(x, self)
        ex = self.exact and self.cfg.rev_backward
        return self.dec(self.run_states(x, ex)[1], ex)

    def _head_loss(self, x, targets):
        logits = self.lm_head(self.ln_f(x)).float()
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), reduction="sum")

    def forward(self, idx, targets):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        x = self.body(x)
        n = self.cfg.ce_chunks
        if n <= 1 or not torch.is_grad_enabled():
            return self._head_loss(x, targets) / targets.numel()
        total = 0.0
        for xc, tc in zip(x.chunk(n, dim=0), targets.chunk(n, dim=0)):
            total = total + checkpoint(self._head_loss, xc, tc, use_reentrant=False)
        return total / targets.numel()
