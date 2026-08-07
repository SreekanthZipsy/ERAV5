"""Checkpoint tied to ledger offset; crash / resume / replay / fork."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .ledgers import LedgerStore
from .model import TinyLM


@dataclass
class CheckpointMeta:
    step: int  # next step to run
    ledger_offset: int
    batch_id_next: str
    batch_hash_next: str | None
    branch: str
    parent_checkpoint: str | None
    tokenizer_hash: str
    model_path: str


class CheckpointStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        name: str,
        model: TinyLM,
        ledgers: LedgerStore,
        meta: CheckpointMeta,
        ledgers_dir: Path,
    ) -> Path:
        ckpt_dir = self.root / name
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        ckpt_dir.mkdir(parents=True)
        model_path = ckpt_dir / "model.npz"
        model.save(model_path, extra={"checkpoint": name, "branch": meta.branch})
        ledgers.save(ckpt_dir / "ledgers")
        # Also mirror to run ledgers dir
        ledgers.save(ledgers_dir)
        meta.model_path = str(model_path)
        (ckpt_dir / "meta.json").write_text(json.dumps(asdict(meta), indent=2, sort_keys=True), encoding="utf-8")
        return ckpt_dir

    def load_meta(self, name: str) -> CheckpointMeta:
        blob = json.loads((self.root / name / "meta.json").read_text(encoding="utf-8"))
        return CheckpointMeta(**blob)

    def restore_model(self, name: str, model: TinyLM) -> CheckpointMeta:
        meta = self.load_meta(name)
        model.load(self.root / name / "model.npz")
        return meta

    def restore_ledgers(self, name: str) -> LedgerStore:
        return LedgerStore.load(self.root / name / "ledgers")

    def fork(self, source: str, dest: str, branch: str) -> CheckpointMeta:
        src = self.root / source
        dst = self.root / dest
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        meta = self.load_meta(dest)
        meta.branch = branch
        meta.parent_checkpoint = source
        (dst / "meta.json").write_text(json.dumps(asdict(meta), indent=2, sort_keys=True), encoding="utf-8")
        return meta
