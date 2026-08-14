"""Tiny transformer body shared across modalities."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, d_model: int = 64, n_layer: int = 2, n_head: int = 4, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.blocks = nn.ModuleList([Block(d_model, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x)


class TextLMKronecker(nn.Module):
    """Character LM: each char is a 1-byte token → Kronecker codec → transformer → logits."""

    def __init__(self, text_codec, d_model: int = 64, n_layer: int = 2, n_head: int = 4, vocab: int = 256):
        super().__init__()
        from .codec import ModalityProjector

        self.codec = text_codec
        self.proj = ModalityProjector(text_codec.spec.D, d_model)
        self.pos = nn.Embedding(512, d_model)
        self.body = TinyTransformer(d_model, n_layer, n_head)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def embed_bytes(self, byte_ids: torch.Tensor) -> torch.Tensor:
        # byte_ids: (B, T) each a single byte → κ from 1-byte sequence
        B, T = byte_ids.shape
        kappas = []
        for b in range(B):
            row = []
            for t in range(T):
                raw = bytes([int(byte_ids[b, t].item())])
                row.append(self.codec.encode_bytes(raw))
            kappas.append(row)
        import numpy as np

        k = torch.tensor(np.array(kappas), dtype=torch.float32, device=byte_ids.device)
        return self.proj(k)

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        B, T = byte_ids.shape
        x = self.embed_bytes(byte_ids)
        x = x + self.pos(torch.arange(T, device=byte_ids.device))[None, :, :]
        h = self.body(x)
        return self.head(h)


class TextLMTable(nn.Module):
    def __init__(self, d_model: int = 64, n_layer: int = 2, n_head: int = 4, vocab: int = 256):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(512, d_model)
        self.body = TinyTransformer(d_model, n_layer, n_head)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        B, T = byte_ids.shape
        x = self.tok(byte_ids) + self.pos(torch.arange(T, device=byte_ids.device))[None, :, :]
        return self.head(self.body(x))


class SequenceClassifier(nn.Module):
    """Kronecker or table embeddings → transformer → mean pool → class."""

    def __init__(self, d_model: int, n_class: int, n_layer: int = 2, n_head: int = 4):
        super().__init__()
        self.body = TinyTransformer(d_model, n_layer, n_head)
        self.head = nn.Linear(d_model, n_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        return self.head(h.mean(dim=1))
