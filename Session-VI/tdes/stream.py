"""Deterministic batch stream: mixture → OPUS → pack → batch ids."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .firewall import EvalFirewall
from .mixture import compile_schedule, iter_lane_quota
from .opus import OpusSelector
from .packing import PackedBatch, pack_spans, packing_utilization
from .shards import TokenSpan
from .tokenizer import FrozenTokenizer


PACKING_BY_LANE = {
    "web": "greedy_concat",
    "code": "best_fit",
    "indic": "greedy_concat",
    "stem": "greedy_concat",
    "reasoning": "structure_preserving",
    "agentic": "structure_preserving",
}


@dataclass
class StreamConfig:
    total_steps: int = 12
    microbatch_sequences: int = 2
    seed: int = 7


class BatchStream:
    """Planned, deterministic sample lookup (Megatron-style mental model)."""

    def __init__(
        self,
        train_spans: list[TokenSpan],
        tok: FrozenTokenizer,
        firewall: EvalFirewall,
        config: StreamConfig,
    ):
        self.tok = tok
        self.firewall = firewall
        self.config = config
        self.id_to_split = {sp.doc_id: sp.split for sp in train_spans}
        # Only train spans should be passed in, but double-filter.
        self.spans = firewall.filter_train_spans(train_spans)
        self.by_lane: dict[str, list[TokenSpan]] = defaultdict(list)
        for sp in self.spans:
            self.by_lane[sp.lane].append(sp)
        for lane in self.by_lane:
            self.by_lane[lane].sort(key=lambda s: s.doc_id)
        self.available = set(self.by_lane.keys())
        self.schedule = compile_schedule(config.total_steps, self.available)
        self.opus = OpusSelector(keep_fraction=0.6)
        self._cursors = {lane: 0 for lane in self.available}
        self._planned_batches: list[PackedBatch] | None = None

    def _next_from_lane(self, lane: str) -> TokenSpan | None:
        pool = self.by_lane.get(lane) or []
        if not pool:
            return None
        idx = self._cursors[lane] % len(pool)
        self._cursors[lane] += 1
        return pool[idx]

    def build_batch(self, step: int) -> PackedBatch:
        plan = self.schedule[step]
        weights = plan["weights"]
        seq_len = plan["seq_len"]
        n_seq = self.config.microbatch_sequences

        # Request candidates per lane quota
        lane_slots = list(iter_lane_quota(weights, n_slots=max(4, n_seq * 3)))
        required = {}
        for lane in lane_slots:
            required[lane] = required.get(lane, 0) + 1

        candidates: list[TokenSpan] = []
        for lane, n in required.items():
            for _ in range(max(n, 1)):
                sp = self._next_from_lane(lane)
                if sp:
                    candidates.append(sp)
        # Pad candidates from web if thin
        while len(candidates) < 4:
            sp = self._next_from_lane("web") or self.spans[0]
            candidates.append(sp)

        # OPUS selection with floor requirements (at least 1 indic/code if in weights)
        floor_need = {}
        for lane, fl in (("indic", 1), ("agentic", 1), ("code", 1)):
            if weights.get(lane, 0) > 0 and lane in self.available:
                floor_need[lane] = 1

        accepted, decisions = self.opus.select(candidates, required_lanes=floor_need)

        # Pack by lane policy groups then merge sequences
        sequences = []
        actual_lane_tokens: dict[str, int] = defaultdict(int)
        # Group accepted by packing policy
        by_policy: dict[str, list[TokenSpan]] = defaultdict(list)
        for sp in accepted:
            by_policy[PACKING_BY_LANE.get(sp.lane, "greedy_concat")].append(sp)

        for policy, group in by_policy.items():
            packed = pack_spans(group, self.tok, seq_len, policy)
            sequences.extend(packed)

        # Trim / pad to microbatch size
        if len(sequences) > n_seq:
            sequences = sequences[:n_seq]
        while len(sequences) < n_seq and sequences:
            sequences.append(sequences[0])

        for seq in sequences:
            for lane in seq.lanes:
                actual_lane_tokens[lane] += seq.useful_tokens // max(len(seq.lanes), 1)
            self.firewall.assert_no_eval_in_batch(seq.doc_ids, self.id_to_split)

        batch_id = f"batch_{step:04d}"
        # Stable hash salt from seed
        _ = hashlib.sha256(f"{self.config.seed}:{step}".encode()).hexdigest()
        opus_summary = []
        for d in decisions:
            opus_summary.append(d.to_dict())

        return PackedBatch(
            batch_id=batch_id,
            step=step,
            sequences=sequences,
            stage=plan["stage"],
            planned_lane_weights=weights,
            actual_lane_tokens=dict(actual_lane_tokens),
            opus_decisions=opus_summary,
        )

    def materialize_all(self) -> list[PackedBatch]:
        """Freeze the full planned stream for resume/replay proofs."""
        # Reset cursors/opus for determinism
        self._cursors = {lane: 0 for lane in self.available}
        self.opus = OpusSelector(keep_fraction=0.6)
        batches = [self.build_batch(step) for step in range(self.config.total_steps)]
        self._planned_batches = batches
        return batches

    def get_batch(self, step: int) -> PackedBatch:
        if self._planned_batches is None:
            self.materialize_all()
        assert self._planned_batches is not None
        return self._planned_batches[step]

    def mixture_compliance(self, batches: list[PackedBatch]) -> dict[str, Any]:
        planned = defaultdict(float)
        actual = defaultdict(float)
        for b in batches:
            for k, v in b.planned_lane_weights.items():
                planned[k] += v
            tot = sum(b.actual_lane_tokens.values()) or 1
            for k, v in b.actual_lane_tokens.items():
                actual[k] += v / tot
        n = len(batches) or 1
        planned_avg = {k: planned[k] / n for k in planned}
        actual_avg = {k: actual[k] / n for k in actual}
        return {"planned_avg": planned_avg, "actual_avg": actual_avg, "n_batches": n}

    def packing_report(self, batches: list[PackedBatch]) -> dict[str, Any]:
        seqs = [s for b in batches for s in b.sequences]
        util = packing_utilization(seqs)
        return util
