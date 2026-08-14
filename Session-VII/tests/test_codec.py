"""Unit tests for Kronecker V2 codecs and invariants."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kronecker_v2.codec import AudioCodec, ImageCodec, TextCodec, cosine, encode_pairs
from kronecker_v2.data import make_shape_image, patches_from_image


def test_text_self_cosine():
    c = TextCodec(d_pos=8)
    k = c.encode_str("kronecker")
    assert cosine(k, k) > 0.99


def test_text_typo_closer_than_unrelated():
    c = TextCodec(d_pos=12)
    base = c.encode_str("embedding")
    typo = c.encode_str("embeddng")
    far = c.encode_str("zzzzzzzz")
    assert cosine(base, typo) > cosine(base, far)


def test_image_brightness_vs_shuffle():
    c = ImageCodec(patch=4, n_levels=5)
    # Use a high-contrast diagonal patch so structure is unambiguous
    p = np.eye(4, dtype=np.float64)
    p = np.clip(p + np.random.default_rng(0).normal(0, 0.02, size=p.shape), 0, 1)
    bright = np.clip(p * 0.5 + 0.25, 0, 1)  # affine brightness — ranks preserved
    flat = p.flatten().copy()
    np.random.default_rng(1).shuffle(flat)
    shuffled = flat.reshape(p.shape)
    k = c.encode_patch(p)
    assert cosine(k, c.encode_patch(bright)) > cosine(k, c.encode_patch(shuffled))


def test_audio_same_vs_diff_freq():
    from kronecker_v2.data import make_tone

    c = AudioCodec(frame_len=32, n_levels=64)
    rng = np.random.default_rng(0)
    a = make_tone(440, rng=rng)[:32]
    b = np.clip(a + rng.normal(0, 0.02, size=32), -1, 1)
    d = make_tone(220, rng=rng)[:32]
    k = c.encode_frame(a)
    assert cosine(k, c.encode_frame(b)) > cosine(k, c.encode_frame(d))


def test_sparse_support_size():
    pairs = [(1, 0), (2, 1), (3, 2)]
    k = encode_pairs(pairs, n_atom=8, n_coord=4, z_norm=False)
    assert np.count_nonzero(k) == 3


def test_dims():
    assert TextCodec(d_pos=16).spec.D == 256 * 16
    assert ImageCodec(patch=4, n_levels=5).spec.D == 5 * 4 * 4
    assert AudioCodec(frame_len=32, n_levels=64).spec.D == 64 * 32
