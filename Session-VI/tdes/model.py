"""Tiny numpy LM — enough for real loss, checkpoint, resume (not scale)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .packing import PackedBatch


@dataclass
class ModelState:
    step: int
    vocab_size: int
    dim: int
    embed: np.ndarray  # (V, D)
    w_out: np.ndarray  # (D, V)
    # Optimizer moments (Adam-ish)
    m_embed: np.ndarray
    v_embed: np.ndarray
    m_w: np.ndarray
    v_w: np.ndarray
    lr: float
    rng_state: dict


class TinyLM:
    def __init__(self, vocab_size: int, dim: int = 32, lr: float = 0.05, seed: int = 42):
        self.vocab_size = vocab_size
        self.dim = dim
        self.lr = lr
        self.rng = np.random.default_rng(seed)
        scale = 0.02
        self.embed = self.rng.normal(0, scale, size=(vocab_size, dim)).astype(np.float64)
        self.w_out = self.rng.normal(0, scale, size=(dim, vocab_size)).astype(np.float64)
        self.m_embed = np.zeros_like(self.embed)
        self.v_embed = np.zeros_like(self.embed)
        self.m_w = np.zeros_like(self.w_out)
        self.v_w = np.zeros_like(self.w_out)
        self.step = 0
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=-1, keepdims=True)

    def train_batch(self, batch: PackedBatch) -> tuple[float, list[float]]:
        """Next-token CE on loss_mask positions. Returns mean_loss, per-token losses."""
        all_token_losses: list[float] = []
        total_loss = 0.0
        n = 0
        grad_embed = np.zeros_like(self.embed)
        grad_w = np.zeros_like(self.w_out)

        for seq in batch.sequences:
            ids = np.array(seq.input_ids, dtype=np.int64)
            loss_mask = np.array(seq.loss_mask, dtype=np.float64)
            attn = np.array(seq.attention_mask, dtype=np.float64)
            # Predict token t from embedding of token t-1 for t>=1 where loss_mask[t]==1
            for t in range(1, len(ids)):
                if loss_mask[t] < 0.5 or attn[t] < 0.5:
                    continue
                if attn[t - 1] < 0.5:
                    continue
                prev = ids[t - 1]
                target = ids[t]
                h = self.embed[prev]  # (D,)
                logits = h @ self.w_out  # (V,)
                probs = self._softmax(logits[None, :])[0]
                loss = float(-np.log(probs[target] + 1e-12))
                all_token_losses.append(loss)
                total_loss += loss
                n += 1
                # dL/dlogits
                dlogits = probs
                dlogits[target] -= 1.0
                grad_w += np.outer(h, dlogits)
                grad_embed[prev] += self.w_out @ dlogits

        if n == 0:
            return 0.0, []

        mean_loss = total_loss / n
        # Adam update
        self.step += 1
        for param, grad, m, v in (
            (self.embed, grad_embed / n, self.m_embed, self.v_embed),
            (self.w_out, grad_w / n, self.m_w, self.v_w),
        ):
            m[:] = self.beta1 * m + (1 - self.beta1) * grad
            v[:] = self.beta2 * v + (1 - self.beta2) * (grad * grad)
            mhat = m / (1 - self.beta1**self.step)
            vhat = v / (1 - self.beta2**self.step)
            param -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

        return mean_loss, all_token_losses

    def state_dict(self) -> ModelState:
        return ModelState(
            step=self.step,
            vocab_size=self.vocab_size,
            dim=self.dim,
            embed=self.embed.copy(),
            w_out=self.w_out.copy(),
            m_embed=self.m_embed.copy(),
            v_embed=self.v_embed.copy(),
            m_w=self.m_w.copy(),
            v_w=self.v_w.copy(),
            lr=self.lr,
            rng_state=self.rng.bit_generator.state,
        )

    def load_state(self, state: ModelState) -> None:
        self.step = state.step
        self.embed = state.embed.copy()
        self.w_out = state.w_out.copy()
        self.m_embed = state.m_embed.copy()
        self.v_embed = state.v_embed.copy()
        self.m_w = state.m_w.copy()
        self.v_w = state.v_w.copy()
        self.lr = state.lr
        self.rng.bit_generator.state = state.rng_state

    def save(self, path: Path, extra: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        st = self.state_dict()
        np.savez_compressed(
            path,
            step=st.step,
            vocab_size=st.vocab_size,
            dim=st.dim,
            embed=st.embed,
            w_out=st.w_out,
            m_embed=st.m_embed,
            v_embed=st.v_embed,
            m_w=st.m_w,
            v_w=st.v_w,
            lr=st.lr,
        )
        meta = {"rng_state": _jsonable_rng(st.rng_state), **(extra or {})}
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def load(self, path: Path) -> dict:
        path = Path(path)
        data = np.load(path)
        self.step = int(data["step"])
        self.vocab_size = int(data["vocab_size"])
        self.dim = int(data["dim"])
        self.embed = data["embed"]
        self.w_out = data["w_out"]
        self.m_embed = data["m_embed"]
        self.v_embed = data["v_embed"]
        self.m_w = data["m_w"]
        self.v_w = data["v_w"]
        self.lr = float(data["lr"])
        meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        if "rng_state" in meta:
            self.rng.bit_generator.state = _restore_rng(meta["rng_state"])
        return meta


def _jsonable_rng(state) -> dict:
    # numpy BitGenerator state -> JSON
    out = {}
    for k, v in state.items():
        if isinstance(v, np.ndarray):
            out[k] = {"__ndarray__": True, "dtype": str(v.dtype), "data": v.tolist()}
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def _restore_rng(meta: dict):
    state = {}
    for k, v in meta.items():
        if isinstance(v, dict) and v.get("__ndarray__"):
            state[k] = np.array(v["data"], dtype=v["dtype"])
        else:
            state[k] = v
    return state
