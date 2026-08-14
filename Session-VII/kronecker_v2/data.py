"""Synthetic multimodal datasets for controlled proofs."""

from __future__ import annotations

import numpy as np


TEXT_CORPUS = (
    "the sun rises in the east and sets in the west. "
    "kronecker embeddings factor bytes and positions. "
    "models learn from data mixtures and curricula. "
    "india has many languages and rich culture. "
    "transformers attend to tokens across long context. "
    "audio waves and image patches share atom coordinate structure. "
    "byte level locality helps spelling robustness a lot. "
    "the quick brown fox jumps over the lazy dog again. "
)


def make_text_windows(seq_len: int = 32, n: int = 512, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Character-level next-token pairs from a tiny corpus (byte ids 0-255)."""
    rng = np.random.default_rng(seed)
    raw = TEXT_CORPUS.encode("utf-8")
    xs, ys = [], []
    for _ in range(n):
        if len(raw) <= seq_len + 1:
            start = 0
        else:
            start = int(rng.integers(0, len(raw) - seq_len - 1))
        window = raw[start : start + seq_len + 1]
        xs.append(list(window[:-1]))
        ys.append(list(window[1:]))
    return np.array(xs, dtype=np.int64), np.array(ys, dtype=np.int64)


def make_shape_image(kind: int, size: int = 8, rng: np.random.Generator | None = None) -> np.ndarray:
    """Synthetic 8×8 shapes: 0=empty, 1=vertical bar, 2=horizontal, 3=diag, 4=box."""
    rng = rng or np.random.default_rng()
    img = np.zeros((size, size), dtype=np.float64)
    noise = rng.normal(0, 0.05, size=(size, size))
    if kind == 1:
        img[:, size // 2] = 1.0
    elif kind == 2:
        img[size // 2, :] = 1.0
    elif kind == 3:
        for i in range(size):
            img[i, i] = 1.0
    elif kind == 4:
        img[1:-1, 1] = 1.0
        img[1:-1, -2] = 1.0
        img[1, 1:-1] = 1.0
        img[-2, 1:-1] = 1.0
    img = np.clip(img + noise, 0, 1)
    return img


def patches_from_image(img: np.ndarray, patch: int = 4) -> list[np.ndarray]:
    H, W = img.shape
    out = []
    for r in range(0, H, patch):
        for c in range(0, W, patch):
            out.append(img[r : r + patch, c : c + patch].copy())
    return out


def make_image_dataset(n: int = 800, patch: int = 4, seed: int = 1):
    rng = np.random.default_rng(seed)
    X_patches, y = [], []
    for _ in range(n):
        kind = int(rng.integers(0, 5))
        img = make_shape_image(kind, size=8, rng=rng)
        pats = patches_from_image(img, patch=patch)
        X_patches.append(pats)
        y.append(kind)
    return X_patches, np.array(y, dtype=np.int64)


def make_tone(freq_hz: float, sr: int = 1600, dur: float = 0.08, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    t = np.arange(int(sr * dur)) / sr
    wave = 0.7 * np.sin(2 * np.pi * freq_hz * t)
    wave += 0.05 * rng.normal(size=wave.shape)
    return np.clip(wave, -1, 1)


def frames_from_wave(wave: np.ndarray, frame_len: int = 32, hop: int = 32) -> list[np.ndarray]:
    out = []
    for i in range(0, len(wave) - frame_len + 1, hop):
        out.append(wave[i : i + frame_len].copy())
    if not out:
        out.append(np.pad(wave, (0, max(0, frame_len - len(wave))))[:frame_len])
    return out


FREQ_CLASSES = [220.0, 330.0, 440.0, 554.0, 660.0]  # A3..E5-ish


def make_audio_dataset(n: int = 800, frame_len: int = 32, seed: int = 2):
    rng = np.random.default_rng(seed)
    X_frames, y = [], []
    for _ in range(n):
        cls = int(rng.integers(0, len(FREQ_CLASSES)))
        # slight detune
        f = FREQ_CLASSES[cls] * (1.0 + float(rng.normal(0, 0.01)))
        wave = make_tone(f, rng=rng)
        frames = frames_from_wave(wave, frame_len=frame_len, hop=frame_len)
        # keep first 4 frames
        while len(frames) < 4:
            frames.append(frames[-1])
        X_frames.append(frames[:4])
        y.append(cls)
    return X_frames, np.array(y, dtype=np.int64)
