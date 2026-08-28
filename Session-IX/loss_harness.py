"""
Session IX — Loss harness helpers (also inlined in the notebook).
Small causal LM + observable CE / packing / chunked CE / MTP t+2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Tiny char tokenizer — strings are readable; off-by-ones show up immediately
# ---------------------------------------------------------------------------

SPECIAL = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<doc>": 3,  # document boundary marker (optional visual)
}


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.id_to_token = ["<pad>", "<bos>", "<eos>", "<doc>"] + [
            c for c in chars if c not in ("\x00",)
        ]
        # dedupe if specials somehow in chars
        seen = set()
        uniq = []
        for t in self.id_to_token:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        self.id_to_token = uniq
        self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}
        self.pad_id = self.token_to_id["<pad>"]
        self.bos_id = self.token_to_id["<bos>"]
        self.eos_id = self.token_to_id["<eos>"]
        self.doc_id = self.token_to_id["<doc>"]
        self.vocab_size = len(self.id_to_token)

    def encode(self, s: str, add_bos=True, add_eos=True) -> List[int]:
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        for c in s:
            ids.append(self.token_to_id.get(c, self.pad_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode_ids(self, ids) -> List[str]:
        out = []
        for i in ids:
            i = int(i)
            if i < 0 or i >= self.vocab_size:
                out.append("<?>")
            else:
                t = self.id_to_token[i]
                if t == " ":
                    out.append("␣")
                elif t == "\n":
                    out.append("↵")
                else:
                    out.append(t)
        return out

    def decode(self, ids) -> str:
        return "".join(
            self.id_to_token[int(i)] if 0 <= int(i) < self.vocab_size else "?"
            for i in ids
        )


# ---------------------------------------------------------------------------
# Tiny GPT
# ---------------------------------------------------------------------------


@dataclass
class GPTConfig:
    vocab_size: int
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    block_size: int = 128
    dropout: float = 0.0
    tie_weights: bool = True
    use_mtp: bool = False  # second head predicts t+2


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Part 2: second head predicts token at t+2
        self.mtp_head = (
            nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False) if cfg.use_mtp else None
        )
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        # GPT-2 residual scaling
        for pn, p in self.named_parameters():
            if pn.endswith("proj.weight") or pn.endswith("mlp.2.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        tokens: (B, T) token ids
        returns hidden (B,T,C) and logits_t1 (B,T,V) [and logits_t2 if MTP]
        """
        B, T = tokens.shape
        assert T <= self.cfg.block_size
        pos = torch.arange(T, device=tokens.device)
        x = self.tok_emb(tokens) + self.pos_emb(pos)[None, :, :]
        for blk in self.blocks:
            x = blk(x)
        hidden = self.ln_f(x)
        logits_t1 = self.lm_head(hidden)
        out = {"hidden": hidden, "logits_t1": logits_t1}
        if self.mtp_head is not None:
            out["logits_t2"] = self.mtp_head(hidden)
        return out


def count_params(model: nn.Module, tied: bool) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    # unique storage: if tied, lm_head.weight is tok_emb.weight
    unique = sum({id(p): p.numel() for p in model.parameters()}.values())
    emb = model.tok_emb.weight.numel()
    head = model.lm_head.weight.numel()
    return {
        "numel_sum_parameters": total,
        "unique_storage": unique,
        "embedding": emb,
        "lm_head_matrix": head,
        "tied": int(tied),
        "emb_plus_head_if_untied": emb + head,
        "emb_plus_head_if_tied": emb,  # shared
    }


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------


def shift_ce_loss(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Standard next-token CE:
      logits[:, :-1] vs tokens[:, 1:]
    Returns (mean_loss, per_position_loss with ignore as nan for viz)
    """
    # logits: (B, T, V), tokens: (B, T)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = tokens[:, 1:].contiguous()
    V = shift_logits.size(-1)
    loss = F.cross_entropy(
        shift_logits.reshape(-1, V),
        shift_labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="mean",
    )
    return loss, shift_logits, shift_labels


def masked_ce_from_logits(
    logits_flat: torch.Tensor,
    labels_flat: torch.Tensor,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, int]:
    """Mean CE over non-ignored labels. Returns (loss, n_contributing)."""
    valid = labels_flat != ignore_index
    n = int(valid.sum().item())
    if n == 0:
        return logits_flat.new_zeros(()), 0
    loss = F.cross_entropy(logits_flat[valid], labels_flat[valid], reduction="mean")
    return loss, n


def perplexity_from_loss(loss: torch.Tensor) -> float:
    return float(math.exp(min(float(loss.item()), 50.0)))  # cap for display


def chunked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Same math as full CE mean over valid tokens, but score in chunks along the
    flattened token axis so peak activation memory for CE stays smaller.
    logits: (N, V) already shifted/flattened OR (B, T, V) with labels (B, T)
    """
    if logits.dim() == 3:
        V = logits.size(-1)
        logits = logits.reshape(-1, V)
        labels = labels.reshape(-1)

    N, V = logits.shape
    total_loss = logits.new_zeros(())
    total_n = 0
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk_logits = logits[start:end]
        chunk_labels = labels[start:end]
        valid = chunk_labels != ignore_index
        n = int(valid.sum().item())
        if n == 0:
            continue
        # sum of per-token NLL, then we'll divide by total_n
        chunk_loss = F.cross_entropy(
            chunk_logits[valid], chunk_labels[valid], reduction="sum"
        )
        total_loss = total_loss + chunk_loss
        total_n += n
    if total_n == 0:
        return total_loss
    return total_loss / total_n


def peak_memory_bytes(fn) -> Tuple[object, int]:
    """Run fn(); return (result, peak allocated bytes). CPU: use cuda if avail else rss approx via tensor nbytes tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        out = fn()
        peak = torch.cuda.max_memory_allocated()
        return out, peak
    # CPU fallback: measure max live tensor bytes created inside by wrapping
    # We approximate by running and reporting activation tensor nbytes of the CE path.
    # For a fair CPU compare we allocate and track with a simple peak counter.
    peak = {"v": 0}

    def _track(t: torch.Tensor):
        peak["v"] = max(peak["v"], t.numel() * t.element_size())
        return t

    out = fn(_track) if _takes_track(fn) else fn()
    # If fn doesn't use tracker, estimate from out tensors
    if peak["v"] == 0:
        # re-run pattern: caller should pass trackable fn
        pass
    return out, peak["v"]


def _takes_track(fn) -> bool:
    import inspect

    return "track" in inspect.signature(fn).parameters


def measure_ce_memory(
    hidden: torch.Tensor,
    head: nn.Linear,
    labels: torch.Tensor,
    ignore_index: int,
    chunk_size: int,
) -> Dict[str, float]:
    """
    Compare peak activation memory of:
      A) full logits = head(hidden) then CE
      B) chunked: project+CE per chunk of positions
    On CUDA uses max_memory_allocated. On CPU measures peak logit tensor bytes.
    """
    device = hidden.device
    use_cuda = device.type == "cuda"

    def full_path():
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(device)
        logits = head(hidden)  # (B, T, V)
        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, V),
            labels[:, 1:].reshape(-1),
            ignore_index=ignore_index,
        )
        if use_cuda:
            peak = torch.cuda.max_memory_allocated(device)
        else:
            peak = logits.numel() * logits.element_size()
        return loss.detach(), peak

    def chunk_path():
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(device)
        B, T, C = hidden.shape
        V = head.out_features
        # shift like CE
        h = hidden[:, :-1, :].reshape(-1, C)  # (N, C)
        y = labels[:, 1:].reshape(-1)
        N = h.size(0)
        total = hidden.new_zeros(())
        n_tok = 0
        peak_logits = 0
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            logits_c = head(h[start:end])  # (chunk, V)
            peak_logits = max(peak_logits, logits_c.numel() * logits_c.element_size())
            valid = y[start:end] != ignore_index
            n = int(valid.sum().item())
            if n:
                total = total + F.cross_entropy(
                    logits_c[valid], y[start:end][valid], reduction="sum"
                )
                n_tok += n
            del logits_c
        loss = total / max(n_tok, 1)
        if use_cuda:
            peak = torch.cuda.max_memory_allocated(device)
        else:
            peak = peak_logits
        return loss.detach(), peak

    loss_a, peak_a = full_path()
    loss_b, peak_b = chunk_path()
    return {
        "full_ce_loss": float(loss_a),
        "chunked_ce_loss": float(loss_b),
        "full_peak_bytes": float(peak_a),
        "chunked_peak_bytes": float(peak_b),
        "ratio_full_over_chunked": float(peak_a) / max(float(peak_b), 1.0),
        "chunk_size": chunk_size,
        "device": str(device),
    }


# ---------------------------------------------------------------------------
# Packing / padding utilities
# ---------------------------------------------------------------------------


def pad_batch(seqs: List[List[int]], pad_id: int) -> torch.Tensor:
    T = max(len(s) for s in seqs)
    B = len(seqs)
    out = torch.full((B, T), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def labels_with_pad_ignored(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Copy tokens; positions that are pad become ignore_index for CE targets.
    For causal LM we ignore when the *target* (tokens[:,1:]) is pad.
    Easiest: set pad positions in the label tensor to -100.
    """
    labels = tokens.clone()
    labels[labels == pad_id] = -100
    return labels


def pack_two_docs(
    doc_a: List[int], doc_b: List[int], pad_id: int, block_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Pack A then B into one sequence (truncate to block_size).
    Returns tokens (1,T), labels_unmasked, labels_boundary_masked, boundary_index
    Boundary mask: at the last position of doc A, the next-token target is the
    first token of doc B — that cross-doc prediction is set to ignore (-100).
    """
    packed = (doc_a + doc_b)[:block_size]
    boundary = len(doc_a) - 1  # position whose target is first token of B
    tokens = torch.tensor([packed], dtype=torch.long)
    labels = tokens.clone()
    labels_masked = tokens.clone()
    # ignore pad if any
    labels[labels == pad_id] = -100
    labels_masked[labels_masked == pad_id] = -100
    # position `boundary` predicts tokens[boundary+1] which is start of doc B
    if 0 <= boundary < tokens.size(1) - 1:
        # In shift CE we use labels[:, 1:], so we mark labels at index boundary+1
        # wait: shift_labels = tokens[:, 1:]; to ignore prediction from pos boundary
        # we need shift_labels[boundary] ignored = tokens[boundary+1] ignored
        # so set labels[boundary+1] = -100  (the target token id slot)
        labels_masked[0, boundary + 1] = -100
    return tokens, labels, labels_masked, boundary


def mtp_losses(
    logits_t1: torch.Tensor,
    logits_t2: torch.Tensor,
    tokens: torch.Tensor,
    ignore_index: int = -100,
) -> Dict[str, torch.Tensor]:
    """
    Head1: predict t+1 from position t  → logits[:, :-1] vs tokens[:, 1:]
    Head2: predict t+2 from position t  → logits_t2[:, :-2] vs tokens[:, 2:]
    """
    V = logits_t1.size(-1)
    loss1 = F.cross_entropy(
        logits_t1[:, :-1].reshape(-1, V),
        tokens[:, 1:].reshape(-1),
        ignore_index=ignore_index,
    )
    loss2 = F.cross_entropy(
        logits_t2[:, :-2].reshape(-1, V),
        tokens[:, 2:].reshape(-1),
        ignore_index=ignore_index,
    )
    return {"loss_t1": loss1, "loss_t2": loss2, "loss_sum": loss1 + loss2}
