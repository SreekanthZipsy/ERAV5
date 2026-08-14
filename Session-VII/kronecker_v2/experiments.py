"""Locality probes + training experiments that prove V2."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .codec import AudioCodec, ImageCodec, TextCodec, UnifiedKroneckerEmbedding, cosine
from .data import (
    FREQ_CLASSES,
    make_audio_dataset,
    make_image_dataset,
    make_shape_image,
    make_text_windows,
    make_tone,
    patches_from_image,
)
from .model import SequenceClassifier, TextLMKronecker, TextLMTable


def locality_text(codec: TextCodec) -> dict:
    pairs = [
        ("run", "runs"),
        ("run", "Run"),
        ("run", "ru"),
        ("compute", "commute"),
        ("nation", "notion"),
        ("kronecker", "kronecker"),
        ("hello", "hallo"),
        ("india", "indica"),
    ]
    rows = []
    for a, b in pairs:
        rows.append({"a": a, "b": b, "cosine": cosine(codec.encode_str(a), codec.encode_str(b))})
    # typo robustness: average clean/typo cosine
    typos = [("embedding", "embeddng"), ("transformer", "transfrmer"), ("language", "langauge")]
    typo_cos = [cosine(codec.encode_str(a), codec.encode_str(b)) for a, b in typos]
    return {"pairs": rows, "mean_typo_cosine": float(np.mean(typo_cos))}


def locality_image(codec: ImageCodec, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    # High-contrast structure patch
    p0 = np.eye(codec.patch, dtype=np.float64)
    p0 = np.clip(p0 + rng.normal(0, 0.03, size=p0.shape), 0, 1)
    bright = np.clip(p0 * 0.5 + 0.25, 0, 1)
    flat = p0.flatten().copy()
    rng.shuffle(flat)
    shuffled = flat.reshape(p0.shape)
    other = np.fliplr(np.eye(codec.patch))
    other = np.clip(other + rng.normal(0, 0.03, size=other.shape), 0, 1)
    k0 = codec.encode_patch(p0)
    return {
        "self": cosine(k0, codec.encode_patch(p0)),
        "brightness_shift": cosine(k0, codec.encode_patch(bright)),
        "pixel_shuffle": cosine(k0, codec.encode_patch(shuffled)),
        "different_shape": cosine(k0, codec.encode_patch(other)),
        "claim": "median-relative atoms: brightness affine stays close; shuffle/other shape separate",
    }


def locality_audio(codec: AudioCodec, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    wave = make_tone(440.0, rng=rng)
    frame = wave[: codec.frame_len]
    noisy = np.clip(frame + rng.normal(0, 0.03, size=frame.shape), -1, 1)
    other = make_tone(220.0, rng=rng)[: codec.frame_len]
    phase = np.roll(frame, 3)
    k0 = codec.encode_frame(frame)
    return {
        "self": cosine(k0, codec.encode_frame(frame)),
        "additive_noise": cosine(k0, codec.encode_frame(noisy)),
        "phase_shift": cosine(k0, codec.encode_frame(phase)),
        "different_freq": cosine(k0, codec.encode_frame(other)),
        "claim": "same-pitch noisy frames cluster; different pitch separates",
    }


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def train_text_lm(steps: int = 400, seed: int = 0, d_model: int = 64) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    X, Y = make_text_windows(seq_len=32, n=256, seed=seed)
    X_t = torch.tensor(X, dtype=torch.long, device=device)
    Y_t = torch.tensor(Y, dtype=torch.long, device=device)

    codec = TextCodec(d_pos=8)
    kron = TextLMKronecker(codec, d_model=d_model, n_layer=2, n_head=4).to(device)
    table = TextLMTable(d_model=d_model, n_layer=2, n_head=4).to(device)

    def run(model, name):
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        losses = []
        t0 = time.perf_counter()
        for step in range(steps):
            idx = torch.randint(0, X_t.size(0), (32,), device=device)
            xb, yb = X_t[idx], Y_t[idx]
            logits = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 20 == 0 or step == steps - 1:
                losses.append({"step": step, "loss": float(loss.item())})
        return {
            "name": name,
            "params": count_params(model),
            "final_loss": losses[-1]["loss"],
            "curve": losses,
            "seconds": time.perf_counter() - t0,
        }

    kr = run(kron, "kronecker_v2_text")
    tb = run(table, "learned_table_text")
    return {
        "kronecker": kr,
        "table": tb,
        "param_ratio_table_over_kron": tb["params"] / max(kr["params"], 1),
        "loss_gap_table_minus_kron": tb["final_loss"] - kr["final_loss"],
    }


def _embed_image_batch(codec: ImageCodec, proj: nn.Module, patches_list, device):
    """patches_list: list of list of patches → (B, T, d)."""
    B = len(patches_list)
    T = len(patches_list[0])
    kappas = np.zeros((B, T, codec.spec.D), dtype=np.float32)
    for i, pats in enumerate(patches_list):
        for j, p in enumerate(pats):
            kappas[i, j] = codec.encode_patch(p).astype(np.float32)
    k = torch.tensor(kappas, device=device)
    return proj(k)


def train_image_clf(steps: int = 300, seed: int = 1, d_model: int = 64) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cpu")
    patches, y = make_image_dataset(n=600, patch=4, seed=seed)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    codec = ImageCodec(patch=4, n_levels=5)
    from .codec import ModalityProjector

    # Kronecker path
    proj = ModalityProjector(codec.spec.D, d_model).to(device)
    clf = SequenceClassifier(d_model, n_class=5, n_layer=2, n_head=4).to(device)
    # Table baseline: hash each patch to a discrete id via quantized signature
    n_hash = 512
    table = nn.Embedding(n_hash, d_model).to(device)
    clf_t = SequenceClassifier(d_model, n_class=5, n_layer=2, n_head=4).to(device)

    def patch_hash(p: np.ndarray) -> int:
        q = codec.quantize(p).astype(np.int64).flatten()
        h = 0
        for v in q:
            h = (h * 31 + int(v)) % n_hash
        return h

    hashes = np.array([[patch_hash(p) for p in pats] for pats in patches], dtype=np.int64)
    H_t = torch.tensor(hashes, dtype=torch.long, device=device)

    def train_kron():
        opt = torch.optim.AdamW(list(proj.parameters()) + list(clf.parameters()), lr=3e-3)
        curve = []
        for step in range(steps):
            idx = torch.randint(0, len(patches), (40,), device=device)
            batch_p = [patches[int(i)] for i in idx.cpu().numpy()]
            x = _embed_image_batch(codec, proj, batch_p, device)
            logits = clf(x)
            loss = F.cross_entropy(logits, y_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 20 == 0 or step == steps - 1:
                with torch.no_grad():
                    # accuracy on a slice
                    sl = slice(0, 120)
                    xb = _embed_image_batch(codec, proj, patches[sl], device)
                    pred = clf(xb).argmax(-1)
                    acc = float((pred == y_t[sl]).float().mean())
                curve.append({"step": step, "loss": float(loss.item()), "acc": acc})
        return curve

    def train_table():
        opt = torch.optim.AdamW(list(table.parameters()) + list(clf_t.parameters()), lr=3e-3)
        curve = []
        for step in range(steps):
            idx = torch.randint(0, len(patches), (40,), device=device)
            x = table(H_t[idx])
            logits = clf_t(x)
            loss = F.cross_entropy(logits, y_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 20 == 0 or step == steps - 1:
                with torch.no_grad():
                    sl = slice(0, 120)
                    pred = clf_t(table(H_t[sl])).argmax(-1)
                    acc = float((pred == y_t[sl]).float().mean())
                curve.append({"step": step, "loss": float(loss.item()), "acc": acc})
        return curve

    ck = train_kron()
    ct = train_table()
    return {
        "kronecker": {
            "params": count_params(proj) + count_params(clf),
            "codec_D": codec.spec.D,
            "final_acc": ck[-1]["acc"],
            "final_loss": ck[-1]["loss"],
            "curve": ck,
        },
        "table_hash": {
            "params": count_params(table) + count_params(clf_t),
            "final_acc": ct[-1]["acc"],
            "final_loss": ct[-1]["loss"],
            "curve": ct,
        },
    }


def train_audio_clf(steps: int = 300, seed: int = 2, d_model: int = 64) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cpu")
    frames, y = make_audio_dataset(n=600, frame_len=32, seed=seed)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    codec = AudioCodec(frame_len=32, n_levels=64)
    from .codec import ModalityProjector

    proj = ModalityProjector(codec.spec.D, d_model).to(device)
    clf = SequenceClassifier(d_model, n_class=len(FREQ_CLASSES), n_layer=2, n_head=4).to(device)

    n_hash = 512
    table = nn.Embedding(n_hash, d_model).to(device)
    clf_t = SequenceClassifier(d_model, n_class=len(FREQ_CLASSES), n_layer=2, n_head=4).to(device)

    def frame_hash(fr: np.ndarray) -> int:
        q = codec.quantize(fr)
        h = 0
        for v in q:
            h = (h * 17 + int(v)) % n_hash
        return h

    hashes = np.array([[frame_hash(f) for f in frs] for frs in frames], dtype=np.int64)
    H_t = torch.tensor(hashes, dtype=torch.long, device=device)

    def embed_kron(batch_frames):
        B, T = len(batch_frames), len(batch_frames[0])
        kappas = np.zeros((B, T, codec.spec.D), dtype=np.float32)
        for i, frs in enumerate(batch_frames):
            for j, fr in enumerate(frs):
                kappas[i, j] = codec.encode_frame(fr).astype(np.float32)
        return proj(torch.tensor(kappas, device=device))

    def train_kron():
        opt = torch.optim.AdamW(list(proj.parameters()) + list(clf.parameters()), lr=3e-3)
        curve = []
        for step in range(steps):
            idx = torch.randint(0, len(frames), (40,), device=device)
            batch = [frames[int(i)] for i in idx.cpu().numpy()]
            loss = F.cross_entropy(clf(embed_kron(batch)), y_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 20 == 0 or step == steps - 1:
                with torch.no_grad():
                    sl = slice(0, 120)
                    pred = clf(embed_kron(frames[sl])).argmax(-1)
                    acc = float((pred == y_t[sl]).float().mean())
                curve.append({"step": step, "loss": float(loss.item()), "acc": acc})
        return curve

    def train_table():
        opt = torch.optim.AdamW(list(table.parameters()) + list(clf_t.parameters()), lr=3e-3)
        curve = []
        for step in range(steps):
            idx = torch.randint(0, len(frames), (40,), device=device)
            loss = F.cross_entropy(clf_t(table(H_t[idx])), y_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 20 == 0 or step == steps - 1:
                with torch.no_grad():
                    sl = slice(0, 120)
                    pred = clf_t(table(H_t[sl])).argmax(-1)
                    acc = float((pred == y_t[sl]).float().mean())
                curve.append({"step": step, "loss": float(loss.item()), "acc": acc})
        return curve

    ck, ct = train_kron(), train_table()
    return {
        "kronecker": {
            "params": count_params(proj) + count_params(clf),
            "codec_D": codec.spec.D,
            "final_acc": ck[-1]["acc"],
            "curve": ck,
        },
        "table_hash": {
            "params": count_params(table) + count_params(clf_t),
            "final_acc": ct[-1]["acc"],
            "curve": ct,
        },
    }


def param_accounting(d_model: int = 4096, vocab: int = 131072) -> dict:
    """Paper-style accounting extended to multimodal V2."""
    text = TextCodec(d_pos=32)
    image = ImageCodec(patch=8, n_levels=8)
    audio = AudioCodec(frame_len=64, n_levels=256)
    return {
        "classic_text_table": vocab * d_model,
        "kronecker_text_proj": text.spec.D * d_model,
        "kronecker_image_proj": image.spec.D * d_model,
        "kronecker_audio_proj": audio.spec.D * d_model,
        "text_D": text.spec.D,
        "image_D": image.spec.D,
        "audio_D": audio.spec.D,
        "text_savings_vs_table": 1.0 - (text.spec.D * d_model) / (vocab * d_model),
    }


def run_all(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text_c, img_c, aud_c = TextCodec(d_pos=8), ImageCodec(patch=4, n_levels=5), AudioCodec(frame_len=32, n_levels=64)

    report = {
        "thesis": (
            "Problem: extend Kronecker beyond text so image patches and audio frames "
            "share the same atom×coordinate factorization. "
            "Solution: κ = |S|^{-1/2} Σ φ(a)⊗ψ(c) (images: atom⊗row⊗col), then e = κ W_proj."
        ),
        "problem": (
            "V1 Kronecker embeddings factor text tokens as byte⊗position. "
            "Can the same structured codec idea represent images and audio without "
            "a new giant embedding table per modality?"
        ),
        "solution": (
            "Treat every modality unit as a set of (atom, coordinate) events; "
            "encode with a length-normalized Kronecker sum; project to d_model."
        ),
        "locality": {
            "text": locality_text(text_c),
            "image": locality_image(img_c),
            "audio": locality_audio(aud_c),
        },
        "params_frontier_scale": param_accounting(),
        "train_text": train_text_lm(steps=350, seed=0),
        "train_image": train_image_clf(steps=250, seed=1),
        "train_audio": train_audio_clf(steps=250, seed=2),
    }

    # Unified front-end sanity: three projectors, shared d_model
    specs = {"text": text_c.spec, "image": img_c.spec, "audio": aud_c.spec}
    unified = UnifiedKroneckerEmbedding(specs, d_model=64)
    report["unified_frontend_params"] = unified.n_trainable()
    report["unified_D"] = {k: s.D for k, s in specs.items()}

    (out_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
