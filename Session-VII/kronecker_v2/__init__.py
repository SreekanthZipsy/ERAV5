"""
Kronecker Embeddings V2 — Multimodal Atom×Coordinate Factorization
===================================================================

V1 (Shravan 2026): text tokens as
    κ(b) = L^{-1/2} Σ_p  onehot(byte_p) ⊗ onehot(pos_p)

V2 natural extension: every modality is a set S of (atom, coordinate) events.
    κ(S) = |S|^{-1/2} Σ_{(a,c) ∈ S}  φ(a) ⊗ ψ(c)

Then a learned projection maps κ → d_model. Only the projection (and optional
tiny atom codebook for continuous sensors) is trainable — not a |V|×d table.

Modalities
----------
- Text  : atom = UTF-8 byte,            coord = byte index in token
- Image : atom = quantized cell value,  coord = (row, col) in patch  →  value ⊗ row ⊗ col
          (2D structure via a triple product, still a single sparse Kronecker vector)
- Audio : atom = μ-law quantized sample / mel bin,  coord = time (× optional freq)

The transformer body is unchanged: it only ever sees e ∈ R^{d_model}.
"""

from __future__ import annotations

__version__ = "0.2.0"
__all__ = [
    "kronecker_sparse",
    "TextCodec",
    "ImageCodec",
    "AudioCodec",
    "ModalityProjector",
    "UnifiedKroneckerEmbedding",
]
