"""Immutable tokenized shards + manifests with content hashes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .corpus import Document
from .hashutil import sha256_ints, sha256_json, sha256_text
from .tokenizer import FrozenTokenizer


@dataclass
class TokenSpan:
    doc_id: str
    lane: str
    split: str
    source: str
    quality: float
    token_ids: list[int]
    loss_mask: list[int]  # 1 = loss-bearing, 0 = context/pad/obs/prompt
    token_span_hash: str = ""

    def __post_init__(self) -> None:
        if not self.token_span_hash:
            self.token_span_hash = sha256_ints(self.token_ids + self.loss_mask)


@dataclass
class ShardManifest:
    shard_id: str
    lane: str
    split: str
    n_docs: int
    n_tokens: int
    n_loss_tokens: int
    content_hash: str
    tokenizer_hash: str
    doc_ids: list[str]
    packing_policy: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _apply_loss_mask_policy(text: str, tok: FrozenTokenizer, lane: str) -> tuple[list[int], list[int]]:
    """Build token ids + loss mask according to lane policy.

    - web/code/indic/stem: all content tokens loss-bearing (pretrain CE)
    - agentic/reasoning: prompt & <<OBS>> observations masked; answers/calls loss-bearing
    """
    if lane in ("agentic", "reasoning") and ("<<PROMPT>>" in text or "<<OBS>>" in text):
        return _structured_mask(text, tok)
    ids = tok.encode(text, add_bos=True, add_eos=True)
    mask = [0] + [1] * (len(ids) - 2) + [1]  # bos no-loss optional; eos can bear loss
    mask[0] = 0
    return ids, mask


def _structured_mask(text: str, tok: FrozenTokenizer) -> tuple[list[int], list[int]]:
    ids: list[int] = [tok.spec.bos_id]
    mask: list[int] = [0]
    # Parse simple tags
    import re

    pattern = re.compile(
        r"<<PROMPT>>(.*?)<</PROMPT>>|<<OBS>>(.*?)</OBS>>|<<ANSWER>>(.*?)</ANSWER>>|([^<]+)",
        re.DOTALL,
    )
    for m in pattern.finditer(text):
        if m.group(1) is not None:
            piece, loss = m.group(1), 0
        elif m.group(2) is not None:
            piece, loss = m.group(2), 0
        elif m.group(3) is not None:
            piece, loss = m.group(3), 1
        else:
            piece, loss = m.group(4), 1
        piece = piece.strip()
        if not piece:
            continue
        part = tok.encode(piece, add_bos=False, add_eos=False)
        ids.extend(part)
        mask.extend([loss] * len(part))
    ids.append(tok.spec.eos_id)
    mask.append(1)
    return ids, mask


def tokenize_documents(docs: list[Document], tok: FrozenTokenizer) -> list[TokenSpan]:
    spans: list[TokenSpan] = []
    for d in docs:
        ids, loss = _apply_loss_mask_policy(d.text, tok, d.lane)
        spans.append(
            TokenSpan(
                doc_id=d.doc_id,
                lane=d.lane,
                split=d.split,
                source=d.source,
                quality=d.quality,
                token_ids=ids,
                loss_mask=loss,
            )
        )
    return spans


def write_shards(
    spans: list[TokenSpan],
    tok: FrozenTokenizer,
    out_dir: Path,
    packing_policy_by_lane: dict[str, str],
) -> list[ShardManifest]:
    """Write one immutable shard per (lane, split). Content-addressed."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, str], list[TokenSpan]] = {}
    for s in spans:
        groups.setdefault((s.lane, s.split), []).append(s)

    manifests: list[ShardManifest] = []
    for (lane, split), group in sorted(groups.items()):
        records = []
        doc_ids = []
        n_tokens = 0
        n_loss = 0
        for sp in group:
            records.append(
                {
                    "doc_id": sp.doc_id,
                    "lane": sp.lane,
                    "split": sp.split,
                    "source": sp.source,
                    "quality": sp.quality,
                    "token_ids": sp.token_ids,
                    "loss_mask": sp.loss_mask,
                    "token_span_hash": sp.token_span_hash,
                }
            )
            doc_ids.append(sp.doc_id)
            n_tokens += len(sp.token_ids)
            n_loss += sum(sp.loss_mask)

        payload = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        content_hash = sha256_text(payload)
        shard_id = f"{lane}_{split}_{content_hash[:12]}"
        shard_path = out_dir / f"{shard_id}.jsonl"
        # Immutable write: refuse overwrite with different content
        if shard_path.exists():
            existing = shard_path.read_text(encoding="utf-8")
            if sha256_text(existing.strip() and "\n".join(existing.splitlines()) or existing) != content_hash:
                # Compare line-normalized
                pass
            old_hash = sha256_text(
                json.dumps(
                    [json.loads(line) for line in shard_path.read_text(encoding="utf-8").splitlines() if line],
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
            )
            if old_hash != content_hash:
                raise RuntimeError(f"Refusing to mutate immutable shard {shard_path}")
        else:
            with shard_path.open("w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        policy = packing_policy_by_lane.get(lane, "greedy_concat")
        man = ShardManifest(
            shard_id=shard_id,
            lane=lane,
            split=split,
            n_docs=len(group),
            n_tokens=n_tokens,
            n_loss_tokens=n_loss,
            content_hash=content_hash,
            tokenizer_hash=tok.hash,
            doc_ids=doc_ids,
            packing_policy=policy,
            metadata={"path": str(shard_path.name)},
        )
        manifests.append(man)
        man_path = out_dir / f"{shard_id}.manifest.json"
        man_path.write_text(json.dumps(asdict(man), indent=2, sort_keys=True), encoding="utf-8")

    index = {
        "tokenizer_hash": tok.hash,
        "shards": [asdict(m) for m in manifests],
        "index_hash": sha256_json({"tokenizer_hash": tok.hash, "shard_ids": [m.shard_id for m in manifests]}),
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    return manifests


def load_shard_spans(shard_dir: Path, shard_id: str) -> list[TokenSpan]:
    man = json.loads((shard_dir / f"{shard_id}.manifest.json").read_text(encoding="utf-8"))
    path = shard_dir / man["metadata"]["path"]
    spans = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        rec = json.loads(line)
        spans.append(
            TokenSpan(
                doc_id=rec["doc_id"],
                lane=rec["lane"],
                split=rec["split"],
                source=rec["source"],
                quality=rec["quality"],
                token_ids=rec["token_ids"],
                loss_mask=rec["loss_mask"],
                token_span_hash=rec["token_span_hash"],
            )
        )
    # Verify content hash
    payload = json.dumps(
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    from .hashutil import sha256_text as _h

    if _h(payload) != man["content_hash"]:
        raise ValueError(f"shard content hash mismatch for {shard_id}")
    return spans


def validate_manifests(shard_dir: Path, expected_tokenizer_hash: str) -> list[str]:
    """Return list of PASS messages; raise on failure."""
    events = []
    index = json.loads((shard_dir / "index.json").read_text(encoding="utf-8"))
    if index["tokenizer_hash"] != expected_tokenizer_hash:
        raise ValueError("index tokenizer_hash mismatch")
    events.append("[PASS] tokenizer_hash_verified")
    for man in index["shards"]:
        load_shard_spans(shard_dir, man["shard_id"])
        if man["tokenizer_hash"] != expected_tokenizer_hash:
            raise ValueError(f"manifest tokenizer mismatch {man['shard_id']}")
    events.append("[PASS] manifests_validated")
    return events
