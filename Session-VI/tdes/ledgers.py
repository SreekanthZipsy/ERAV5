"""Consumption ledger (what we fed) and learning ledger (what the model learned)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ConsumptionRecord:
    step: int
    batch_id: str
    batch_hash: str
    stage: str
    doc_ids: list[str]
    lanes: list[str]
    useful_tokens: int
    pad_tokens: int
    token_span_hashes: list[str]
    opus_summary: dict[str, int]
    ledger_offset: int  # append-only index


@dataclass
class LearningRecord:
    step: int
    batch_id: str
    batch_hash: str
    mean_loss: float
    token_losses: list[float]  # per useful token (or truncated sample)
    loss_linked_docs: list[str]
    perplexity: float
    ledger_offset: int


@dataclass
class LedgerStore:
    consumption: list[ConsumptionRecord] = field(default_factory=list)
    learning: list[LearningRecord] = field(default_factory=list)

    @property
    def offset(self) -> int:
        return len(self.consumption)

    def append_consumption(self, rec: ConsumptionRecord) -> None:
        assert rec.ledger_offset == len(self.consumption)
        self.consumption.append(rec)

    def append_learning(self, rec: LearningRecord) -> None:
        assert rec.ledger_offset == len(self.learning)
        self.learning.append(rec)

    def truncate_to_offset(self, offset: int) -> None:
        """Crash recovery: drop speculative records after checkpoint offset."""
        self.consumption = self.consumption[:offset]
        self.learning = self.learning[:offset]

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        cons_path = directory / "consumption.jsonl"
        learn_path = directory / "learning.jsonl"
        with cons_path.open("w", encoding="utf-8") as f:
            for r in self.consumption:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        with learn_path.open("w", encoding="utf-8") as f:
            for r in self.learning:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        meta = {
            "consumption_offset": len(self.consumption),
            "learning_offset": len(self.learning),
        }
        (directory / "ledger_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "LedgerStore":
        directory = Path(directory)
        store = cls()
        cons_path = directory / "consumption.jsonl"
        learn_path = directory / "learning.jsonl"
        if cons_path.exists():
            for line in cons_path.read_text(encoding="utf-8").splitlines():
                if line:
                    store.consumption.append(ConsumptionRecord(**json.loads(line)))
        if learn_path.exists():
            for line in learn_path.read_text(encoding="utf-8").splitlines():
                if line:
                    d = json.loads(line)
                    store.learning.append(LearningRecord(**d))
        return store

    def summary(self) -> dict[str, Any]:
        return {
            "n_consumption": len(self.consumption),
            "n_learning": len(self.learning),
            "steps": [r.step for r in self.consumption],
            "batch_ids": [r.batch_id for r in self.consumption],
            "batch_hashes": [r.batch_hash for r in self.consumption],
        }
