"""Packing policies: pad_only, greedy_concat, best_fit, structure_preserving."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .hashutil import sha256_ints, sha256_json
from .shards import TokenSpan
from .tokenizer import FrozenTokenizer


@dataclass
class PackedSequence:
    input_ids: list[int]
    loss_mask: list[int]
    attention_mask: list[int]
    position_ids: list[int]
    doc_ids: list[str]
    lanes: list[str]
    packing_policy: str
    useful_tokens: int
    pad_tokens: int
    sequence_hash: str = ""

    def __post_init__(self) -> None:
        if not self.sequence_hash:
            self.sequence_hash = sha256_ints(self.input_ids + self.loss_mask + self.position_ids)


@dataclass
class PackedBatch:
    batch_id: str
    step: int
    sequences: list[PackedSequence]
    stage: str
    planned_lane_weights: dict[str, float]
    actual_lane_tokens: dict[str, int]
    opus_decisions: list[dict[str, Any]] = field(default_factory=list)
    batch_hash: str = ""

    def __post_init__(self) -> None:
        if not self.batch_hash:
            self.batch_hash = sha256_json(
                {
                    "batch_id": self.batch_id,
                    "step": self.step,
                    "seq_hashes": [s.sequence_hash for s in self.sequences],
                    "stage": self.stage,
                }
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "step": self.step,
            "stage": self.stage,
            "batch_hash": self.batch_hash,
            "planned_lane_weights": self.planned_lane_weights,
            "actual_lane_tokens": self.actual_lane_tokens,
            "opus_decisions": self.opus_decisions,
            "sequences": [asdict(s) for s in self.sequences],
            "n_sequences": len(self.sequences),
            "useful_tokens": sum(s.useful_tokens for s in self.sequences),
            "pad_tokens": sum(s.pad_tokens for s in self.sequences),
        }


def pack_spans(
    spans: list[TokenSpan],
    tok: FrozenTokenizer,
    seq_len: int,
    policy: str,
) -> list[PackedSequence]:
    if policy == "pad_only":
        return [_pad_one(sp, tok, seq_len, policy) for sp in spans]
    if policy == "structure_preserving":
        # Never concatenate unrelated agent/reasoning traces.
        return [_pad_one(sp, tok, seq_len, policy) for sp in spans]
    if policy == "best_fit":
        return _best_fit(spans, tok, seq_len)
    # default greedy_concat
    return _greedy_concat(spans, tok, seq_len)


def _pad_one(sp: TokenSpan, tok: FrozenTokenizer, seq_len: int, policy: str) -> PackedSequence:
    ids = list(sp.token_ids[:seq_len])
    mask = list(sp.loss_mask[:seq_len])
    if len(ids) < seq_len:
        pad_n = seq_len - len(ids)
        ids.extend([tok.spec.pad_id] * pad_n)
        mask.extend([0] * pad_n)
    attn = [0 if i == tok.spec.pad_id else 1 for i in ids]
    # Position ids restart after EOS for concatenated packs; single doc: 0..n
    pos = list(range(seq_len))
    useful = sum(1 for a, m in zip(attn, mask) if a and m)
    pad_tokens = sum(1 for i in ids if i == tok.spec.pad_id)
    return PackedSequence(
        input_ids=ids,
        loss_mask=mask,
        attention_mask=attn,
        position_ids=pos,
        doc_ids=[sp.doc_id],
        lanes=[sp.lane],
        packing_policy=policy,
        useful_tokens=useful,
        pad_tokens=pad_tokens,
    )


def _greedy_concat(spans: list[TokenSpan], tok: FrozenTokenizer, seq_len: int) -> list[PackedSequence]:
    out: list[PackedSequence] = []
    cur_ids: list[int] = []
    cur_mask: list[int] = []
    cur_pos: list[int] = []
    cur_docs: list[str] = []
    cur_lanes: list[str] = []
    pos = 0

    def flush() -> None:
        nonlocal cur_ids, cur_mask, cur_pos, cur_docs, cur_lanes, pos
        if not cur_ids:
            return
        pad_n = seq_len - len(cur_ids)
        ids = cur_ids + [tok.spec.pad_id] * pad_n
        mask = cur_mask + [0] * pad_n
        pos_ids = cur_pos + [0] * pad_n
        attn = [0 if i == tok.spec.pad_id else 1 for i in ids]
        useful = sum(1 for a, m in zip(attn, mask) if a and m)
        out.append(
            PackedSequence(
                input_ids=ids,
                loss_mask=mask,
                attention_mask=attn,
                position_ids=pos_ids,
                doc_ids=list(cur_docs),
                lanes=list(cur_lanes),
                packing_policy="greedy_concat",
                useful_tokens=useful,
                pad_tokens=pad_n,
            )
        )
        cur_ids, cur_mask, cur_pos, cur_docs, cur_lanes = [], [], [], [], []
        pos = 0

    for sp in spans:
        piece = list(sp.token_ids)
        pmask = list(sp.loss_mask)
        # Ensure document ends with eos already from tokenize
        if len(piece) > seq_len:
            piece = piece[:seq_len]
            pmask = pmask[:seq_len]
        if len(cur_ids) + len(piece) > seq_len:
            flush()
        # position restart after packing a new doc into window
        for i, (tid, m) in enumerate(zip(piece, pmask)):
            cur_ids.append(tid)
            cur_mask.append(m)
            cur_pos.append(pos)
            pos += 1
            if tid == tok.spec.eos_id:
                pos = 0  # context switch
        cur_docs.append(sp.doc_id)
        cur_lanes.append(sp.lane)
    flush()
    return out


def _best_fit(spans: list[TokenSpan], tok: FrozenTokenizer, seq_len: int) -> list[PackedSequence]:
    """Sort by length descending, place into first bin with room (best-fit decreasing)."""
    items = sorted(spans, key=lambda s: len(s.token_ids), reverse=True)
    bins: list[list[TokenSpan]] = []
    bin_fill: list[int] = []
    for sp in items:
        L = min(len(sp.token_ids), seq_len)
        best_i = -1
        best_remain = 10**9
        for i, fill in enumerate(bin_fill):
            remain = seq_len - fill
            if L <= remain and remain - L < best_remain:
                best_remain = remain - L
                best_i = i
        if best_i < 0:
            bins.append([sp])
            bin_fill.append(L)
        else:
            bins[best_i].append(sp)
            bin_fill[best_i] += L
    out: list[PackedSequence] = []
    for group in bins:
        packed = _greedy_concat(group, tok, seq_len)
        for p in packed:
            p.packing_policy = "best_fit"
        out.extend(packed)
    return out


def packing_utilization(sequences: list[PackedSequence]) -> dict[str, float]:
    useful = sum(s.useful_tokens for s in sequences)
    total = sum(len(s.input_ids) for s in sequences)
    pad = sum(s.pad_tokens for s in sequences)
    return {
        "useful_tokens": float(useful),
        "total_slots": float(total),
        "pad_tokens": float(pad),
        "utilization": (useful / total) if total else 0.0,
        "pad_fraction": (pad / total) if total else 0.0,
    }
