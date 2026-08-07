"""OPUS-style candidate selection: accept / reject / defer / protected-floor override.

Demo uses a cheap proxy score (quality * novelty vs recent) rather than real gradients.
Decisions are fully audited.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .mixture import PROTECTED_FLOORS
from .shards import TokenSpan


@dataclass
class OpusDecision:
    doc_id: str
    lane: str
    decision: str  # accept | reject | defer | floor_override
    score: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "lane": self.lane,
            "decision": self.decision,
            "score": self.score,
            "reason": self.reason,
        }


@dataclass
class OpusSelector:
    keep_fraction: float = 0.55
    probe_tokens: int = 16
    recent_hashes: set[str] = field(default_factory=set)
    decisions: list[OpusDecision] = field(default_factory=list)

    def score_span(self, span: TokenSpan) -> float:
        # Proxy for "updates weak weights": high quality + not recently seen.
        novelty = 0.0 if span.token_span_hash in self.recent_hashes else 1.0
        # First probe_tokens stand in for the 512-token OPUS probe.
        probe = span.token_ids[: self.probe_tokens]
        diversity = len(set(probe)) / max(len(probe), 1)
        return float(span.quality) * 0.6 + novelty * 0.25 + diversity * 0.15

    def select(
        self,
        candidates: list[TokenSpan],
        required_lanes: dict[str, int],
    ) -> tuple[list[TokenSpan], list[OpusDecision]]:
        """Select spans; ensure required_lanes counts via floor_override if needed."""
        scored = [(self.score_span(sp), sp) for sp in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        n_keep = max(1, int(len(scored) * self.keep_fraction))
        threshold = scored[n_keep - 1][0] if scored else 0.0

        accepted: list[TokenSpan] = []
        decisions: list[OpusDecision] = []
        deferred_pool: list[TokenSpan] = []

        for rank, (score, sp) in enumerate(scored):
            if rank < n_keep:
                if sp.token_span_hash in self.recent_hashes and score < 0.95:
                    dec = OpusDecision(sp.doc_id, sp.lane, "defer", score, "recently_seen_defer")
                    deferred_pool.append(sp)
                elif sp.quality < 0.55:
                    dec = OpusDecision(sp.doc_id, sp.lane, "defer", score, "quality_below_floor")
                    deferred_pool.append(sp)
                else:
                    dec = OpusDecision(sp.doc_id, sp.lane, "accept", score, "proxy_affinity")
                    accepted.append(sp)
                    self.recent_hashes.add(sp.token_span_hash)
            elif score >= threshold * 0.9:
                dec = OpusDecision(sp.doc_id, sp.lane, "defer", score, "borderline_probe")
                deferred_pool.append(sp)
            else:
                dec = OpusDecision(sp.doc_id, sp.lane, "reject", score, "low_proxy_affinity")
            decisions.append(dec)
        # Guarantee at least one reject in the audit trail when candidates are plentiful
        if len(scored) > n_keep and not any(d.decision == "reject" for d in decisions):
            score, sp = scored[-1]
            decisions.append(
                OpusDecision(sp.doc_id, sp.lane, "reject", score, "forced_tail_reject_for_audit")
            )

        # Protected-floor override: if a required lane is missing, force-accept best of that lane.
        have = {}
        for sp in accepted:
            have[sp.lane] = have.get(sp.lane, 0) + 1
        by_lane: dict[str, list[TokenSpan]] = {}
        for sp in candidates:
            by_lane.setdefault(sp.lane, []).append(sp)

        for lane, need in required_lanes.items():
            while have.get(lane, 0) < need:
                pool = [sp for sp in by_lane.get(lane, []) if sp not in accepted]
                if not pool:
                    break
                pool.sort(key=self.score_span, reverse=True)
                forced = pool[0]
                accepted.append(forced)
                have[lane] = have.get(lane, 0) + 1
                self.recent_hashes.add(forced.token_span_hash)
                decisions.append(
                    OpusDecision(
                        forced.doc_id,
                        forced.lane,
                        "floor_override",
                        self.score_span(forced),
                        f"protected_floor:{PROTECTED_FLOORS.get(lane, need)}",
                    )
                )

        self.decisions.extend(decisions)
        return accepted, decisions
