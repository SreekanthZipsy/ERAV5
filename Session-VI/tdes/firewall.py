"""Evaluation / validation firewall — blocks non-train splits from loss batches."""

from __future__ import annotations

from dataclasses import dataclass, field

from .shards import TokenSpan


@dataclass
class FirewallEvent:
    event: str
    shard_or_doc: str
    split: str
    action: str


@dataclass
class EvalFirewall:
    blocked: list[FirewallEvent] = field(default_factory=list)

    def filter_train_spans(self, spans: list[TokenSpan]) -> list[TokenSpan]:
        kept: list[TokenSpan] = []
        for sp in spans:
            if sp.split != "train":
                self.blocked.append(
                    FirewallEvent(
                        event="eval_shard_blocked",
                        shard_or_doc=sp.doc_id,
                        split=sp.split,
                        action="reject_from_loss_batch",
                    )
                )
                continue
            kept.append(sp)
        return kept

    def assert_no_eval_in_batch(self, doc_ids: list[str], id_to_split: dict[str, str]) -> None:
        for d in doc_ids:
            if id_to_split.get(d, "train") != "train":
                raise RuntimeError(f"FIREWALL BREACH: {d} split={id_to_split[d]} entered loss batch")
