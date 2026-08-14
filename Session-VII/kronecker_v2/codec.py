"""Core sparse Kronecker codec: sum of atom ⊗ coordinate one-hots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def kronecker_index(atom: int, coord: int, n_coord: int) -> int:
    """Flat index of onehot(atom) ⊗ onehot(coord) in R^{n_atom * n_coord}."""
    return int(atom) * int(n_coord) + int(coord)


def encode_pairs(
    pairs: Sequence[tuple[int, int]],
    n_atom: int,
    n_coord: int,
    z_norm: bool = True,
) -> np.ndarray:
    """κ = |S|^{-1/2} Σ (a⊗c); optional per-vector z-norm (V1 §3.3)."""
    D = n_atom * n_coord
    kappa = np.zeros(D, dtype=np.float64)
    if not pairs:
        return kappa
    scale = 1.0 / np.sqrt(len(pairs))
    for a, c in pairs:
        a = int(a) % n_atom
        c = int(c) % n_coord
        kappa[kronecker_index(a, c, n_coord)] += scale
    if z_norm:
        mu, sd = kappa.mean(), kappa.std()
        if sd > 1e-8:
            kappa = (kappa - mu) / sd
    return kappa


def encode_triples(
    triples: Sequence[tuple[int, int, int]],
    n_a: int,
    n_b: int,
    n_c: int,
    z_norm: bool = True,
) -> np.ndarray:
    """κ = |S|^{-1/2} Σ onehot(a)⊗onehot(b)⊗onehot(c)  (image 2D)."""
    D = n_a * n_b * n_c
    kappa = np.zeros(D, dtype=np.float64)
    if not triples:
        return kappa
    scale = 1.0 / np.sqrt(len(triples))
    for a, b, c in triples:
        a, b, c = int(a) % n_a, int(b) % n_b, int(c) % n_c
        idx = (a * n_b + b) * n_c + c
        kappa[idx] += scale
    if z_norm:
        mu, sd = kappa.mean(), kappa.std()
        if sd > 1e-8:
            kappa = (kappa - mu) / sd
    return kappa


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    u = u - u.mean()
    v = v - v.mean()
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


@dataclass(frozen=True)
class CodecSpec:
    name: str
    n_atom: int
    n_coord: int
    extra_axes: tuple[int, ...] = ()

    @property
    def D(self) -> int:
        d = self.n_atom * self.n_coord
        for ax in self.extra_axes:
            d *= ax
        return d


class TextCodec:
    """V1-compatible: UTF-8 bytes × byte positions."""

    def __init__(self, d_pos: int = 16, z_norm: bool = True):
        self.spec = CodecSpec("text", n_atom=256, n_coord=d_pos)
        self.d_pos = d_pos
        self.z_norm = z_norm

    def pairs_from_bytes(self, data: bytes) -> list[tuple[int, int]]:
        data = data[: self.d_pos]
        return [(b, p) for p, b in enumerate(data)]

    def encode_bytes(self, data: bytes) -> np.ndarray:
        return encode_pairs(self.pairs_from_bytes(data), 256, self.d_pos, self.z_norm)

    def encode_str(self, s: str) -> np.ndarray:
        return self.encode_bytes(s.encode("utf-8"))


class ImageCodec:
    """
    Patch as set of (structure_atom, row, col).

    Atoms are **median-relative ranks** inside the patch (not raw brightness),
    so the inductive bias is spatial structure — the 2D analogue of V1's
    byte-level locality. Triple Kronecker: atom ⊗ row ⊗ col.
    """

    def __init__(self, patch: int = 4, n_levels: int = 5, z_norm: bool = True):
        # n_levels odd preferred: bins around median
        self.patch = patch
        self.n_levels = n_levels
        self.z_norm = z_norm
        self.spec = CodecSpec("image", n_atom=n_levels, n_coord=patch, extra_axes=(patch,))

    def quantize(self, patch: np.ndarray) -> np.ndarray:
        """Map each cell to a rank bin relative to the patch median."""
        x = patch.astype(np.float64)
        med = np.median(x)
        # signed deviation → percentile-ish bins via tanh scale
        dev = x - med
        scale = np.std(x) + 1e-6
        z = np.tanh(dev / scale)  # (-1, 1)
        u = (z + 1.0) * 0.5  # (0, 1)
        return np.floor(np.clip(u, 0, 0.999999) * self.n_levels).astype(np.int64)

    def triples_from_patch(self, patch: np.ndarray) -> list[tuple[int, int, int]]:
        q = self.quantize(patch)
        H, W = q.shape
        assert H == self.patch and W == self.patch
        out = []
        for r in range(H):
            for c in range(W):
                out.append((int(q[r, c]), r, c))
        return out

    def encode_patch(self, patch: np.ndarray) -> np.ndarray:
        return encode_triples(
            self.triples_from_patch(patch),
            self.n_levels,
            self.patch,
            self.patch,
            self.z_norm,
        )


class AudioCodec:
    """
    Frame as μ-law quantized samples × time index.
    Same factorization as text; atoms are continuous-signal buckets.
    """

    def __init__(self, frame_len: int = 32, n_levels: int = 64, z_norm: bool = True):
        self.frame_len = frame_len
        self.n_levels = n_levels
        self.z_norm = z_norm
        self.spec = CodecSpec("audio", n_atom=n_levels, n_coord=frame_len)

    @staticmethod
    def mulaw(x: np.ndarray, mu: float = 255.0) -> np.ndarray:
        x = np.clip(x, -1.0, 1.0)
        return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)

    def quantize(self, frame: np.ndarray) -> np.ndarray:
        y = self.mulaw(frame.astype(np.float64))
        # map [-1,1] → [0, n_levels-1]
        u = (y + 1.0) * 0.5
        return np.floor(np.clip(u, 0, 1) * (self.n_levels - 1) + 1e-8).astype(np.int64)

    def pairs_from_frame(self, frame: np.ndarray) -> list[tuple[int, int]]:
        frame = frame[: self.frame_len]
        if len(frame) < self.frame_len:
            frame = np.pad(frame, (0, self.frame_len - len(frame)))
        q = self.quantize(frame)
        return [(int(q[t]), t) for t in range(self.frame_len)]

    def encode_frame(self, frame: np.ndarray) -> np.ndarray:
        return encode_pairs(
            self.pairs_from_frame(frame),
            self.n_levels,
            self.frame_len,
            self.z_norm,
        )


class ModalityProjector(nn.Module):
    """Learned W: R^D → R^{d_model}. Only trainable input-side params for a modality."""

    def __init__(self, D: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(D, d_model, bias=False)
        nn.init.normal_(self.proj.weight, std=D**-0.5)

    def forward(self, kappa: torch.Tensor) -> torch.Tensor:
        return self.proj(kappa)


class UnifiedKroneckerEmbedding(nn.Module):
    """
    Shared d_model interface across modalities.
    Precomputes κ offline or accepts dense κ batches.
    """

    def __init__(self, specs: dict[str, CodecSpec], d_model: int):
        super().__init__()
        self.d_model = d_model
        self.projectors = nn.ModuleDict(
            {name: ModalityProjector(spec.D, d_model) for name, spec in specs.items()}
        )
        self.modality_bias = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(d_model)) for name in specs}
        )

    def forward(self, modality: str, kappa: torch.Tensor) -> torch.Tensor:
        return self.projectors[modality](kappa) + self.modality_bias[modality]

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
