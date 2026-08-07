"""Training loop with deliberate crash, resume, replay, fork, audit."""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .checkpoint import CheckpointMeta, CheckpointStore
from .ledgers import ConsumptionRecord, LearningRecord, LedgerStore
from .model import TinyLM
from .packing import PackedBatch
from .stream import BatchStream


LogFn = Callable[[str], None]


class Trainer:
    def __init__(
        self,
        stream: BatchStream,
        model: TinyLM,
        artifacts: Path,
        tokenizer_hash: str,
        log: LogFn,
    ):
        self.stream = stream
        self.model = model
        self.artifacts = Path(artifacts)
        self.tokenizer_hash = tokenizer_hash
        self.log = log
        self.ledgers = LedgerStore()
        self.ckpts = CheckpointStore(self.artifacts / "checkpoints")
        self.ledgers_dir = self.artifacts / "ledgers"
        self.ledgers_dir.mkdir(parents=True, exist_ok=True)
        self.batch_archive: dict[str, dict] = {}
        self.perf: dict[str, Any] = {}

    def _consume(self, batch: PackedBatch, mean_loss: float, token_losses: list[float]) -> None:
        offset = self.ledgers.offset
        opus_counts = Counter(d["decision"] for d in batch.opus_decisions)
        doc_ids = sorted({d for s in batch.sequences for d in s.doc_ids})
        lanes = sorted({ln for s in batch.sequences for ln in s.lanes})
        span_hashes = []
        for s in batch.sequences:
            span_hashes.append(s.sequence_hash)

        self.ledgers.append_consumption(
            ConsumptionRecord(
                step=batch.step,
                batch_id=batch.batch_id,
                batch_hash=batch.batch_hash,
                stage=batch.stage,
                doc_ids=doc_ids,
                lanes=lanes,
                useful_tokens=sum(s.useful_tokens for s in batch.sequences),
                pad_tokens=sum(s.pad_tokens for s in batch.sequences),
                token_span_hashes=span_hashes,
                opus_summary=dict(opus_counts),
                ledger_offset=offset,
            )
        )
        ppl = math.exp(min(mean_loss, 20.0))
        self.ledgers.append_learning(
            LearningRecord(
                step=batch.step,
                batch_id=batch.batch_id,
                batch_hash=batch.batch_hash,
                mean_loss=mean_loss,
                token_losses=token_losses[:256],
                loss_linked_docs=doc_ids,
                perplexity=ppl,
                ledger_offset=offset,
            )
        )
        self.batch_archive[batch.batch_id] = batch.to_dict()

    def run_steps(self, start: int, end: int, crash_at: int | None = None) -> int:
        """Run steps in [start, end). If crash_at is set, stop before that step after saving ckpt."""
        t0 = time.perf_counter()
        useful = 0
        for step in range(start, end):
            if crash_at is not None and step == crash_at:
                # Save checkpoint for resume at crash_at (next batch)
                nxt = self.stream.get_batch(step)
                self._save_ckpt(
                    f"ckpt_pre_crash_step{step}",
                    step=step,
                    next_batch=nxt,
                    branch="main",
                )
                self.log(f"[PASS] checkpoint_saved ckpt_pre_crash_step{step} next={nxt.batch_id}")
                self.log(f"[EVENT] crash_simulated at_step={step}")
                raise RuntimeError(f"SIMULATED_CRASH at step {step}")

            batch = self.stream.get_batch(step)
            self.log(f"[EVENT] batch_packed {batch.batch_id} hash={batch.batch_hash[:12]} stage={batch.stage}")
            for d in batch.opus_decisions:
                if d["decision"] in ("accept", "reject", "defer", "floor_override"):
                    self.log(
                        f"[EVENT] opus_{d['decision']} doc={d['doc_id']} score={d['score']:.3f} reason={d['reason']}"
                    )
            mean_loss, token_losses = self.model.train_batch(batch)
            self._consume(batch, mean_loss, token_losses)
            useful += sum(s.useful_tokens for s in batch.sequences)
            self.log(
                f"[EVENT] trained {batch.batch_id} loss={mean_loss:.4f} useful={sum(s.useful_tokens for s in batch.sequences)}"
            )

            # Periodic checkpoint every 3 steps
            if (step + 1) % 3 == 0 and step + 1 < end:
                nxt_step = step + 1
                nxt = self.stream.get_batch(nxt_step) if nxt_step < self.stream.config.total_steps else None
                self._save_ckpt(
                    f"ckpt_step{nxt_step}",
                    step=nxt_step,
                    next_batch=nxt,
                    branch="main",
                )
                self.log(f"[PASS] checkpoint_saved ckpt_step{nxt_step}")

        elapsed = time.perf_counter() - t0
        self.perf["useful_tokens"] = self.perf.get("useful_tokens", 0) + useful
        self.perf["elapsed_sec"] = self.perf.get("elapsed_sec", 0.0) + elapsed
        return end

    def _save_ckpt(self, name: str, step: int, next_batch: PackedBatch | None, branch: str) -> None:
        meta = CheckpointMeta(
            step=step,
            ledger_offset=self.ledgers.offset,
            batch_id_next=next_batch.batch_id if next_batch else "",
            batch_hash_next=next_batch.batch_hash if next_batch else None,
            branch=branch,
            parent_checkpoint=None,
            tokenizer_hash=self.tokenizer_hash,
            model_path="",
        )
        self.ckpts.save(name, self.model, self.ledgers, meta, self.ledgers_dir)

    def resume_from(self, ckpt_name: str, end: int) -> dict[str, Any]:
        meta = self.ckpts.restore_model(ckpt_name, self.model)
        self.ledgers = self.ckpts.restore_ledgers(ckpt_name)
        expected_id = meta.batch_id_next
        expected_hash = meta.batch_hash_next
        actual = self.stream.get_batch(meta.step)
        matched = actual.batch_id == expected_id and actual.batch_hash == expected_hash
        self.log(
            f"[EVENT] run_resumed from={ckpt_name} next_step={meta.step} expected={expected_id} actual={actual.batch_id}"
        )
        if matched:
            self.log("[PASS] resume_next_batch_matched")
        else:
            self.log(
                f"[FAIL] resume_next_batch_matched expected={expected_id}/{expected_hash} got={actual.batch_id}/{actual.batch_hash}"
            )
        self.run_steps(meta.step, end, crash_at=None)
        return {
            "matched": matched,
            "expected_batch_id": expected_id,
            "actual_batch_id": actual.batch_id,
            "expected_hash": expected_hash,
            "actual_hash": actual.batch_hash,
            "resume_step": meta.step,
        }

    def replay_interval(self, start: int, end: int) -> dict[str, Any]:
        """Rebuild batches for [start, end) and compare to archived hashes."""
        results = []
        all_match = True
        for step in range(start, end):
            rebuilt = self.stream.get_batch(step)
            original = self.batch_archive.get(rebuilt.batch_id)
            if original is None:
                # fall back to consumption ledger
                cons = next((c for c in self.ledgers.consumption if c.step == step), None)
                orig_hash = cons.batch_hash if cons else None
            else:
                orig_hash = original["batch_hash"]
            match = orig_hash == rebuilt.batch_hash
            all_match = all_match and match
            results.append(
                {
                    "step": step,
                    "batch_id": rebuilt.batch_id,
                    "original_hash": orig_hash,
                    "replay_hash": rebuilt.batch_hash,
                    "match": match,
                    "token_spans": [s.sequence_hash for s in rebuilt.sequences],
                }
            )
            self.log(
                f"[EVENT] replay step={step} batch={rebuilt.batch_id} match={match}"
            )
        if all_match:
            self.log("[PASS] replay_hash_matched")
        else:
            self.log("[FAIL] replay_hash_matched")
        return {"all_match": all_match, "results": results}

    def fork_branch(self, source_ckpt: str, branch: str) -> dict[str, Any]:
        import json
        import math
        from collections import Counter
        from dataclasses import asdict

        dest = f"ckpt_fork_{branch}"
        meta = self.ckpts.fork(source_ckpt, dest, branch)
        self.log(f"[EVENT] branch_forked source={source_ckpt} dest={dest} branch={branch}")

        main_ledgers = self.ledgers
        fork_model = TinyLM(self.model.vocab_size, dim=self.model.dim, lr=self.model.lr, seed=0)
        self.ckpts.restore_model(dest, fork_model)
        fork_ledgers = self.ckpts.restore_ledgers(dest)
        step = meta.step
        fork_batch_id = None
        after_dir = self.ckpts.root / f"ckpt_{branch}_after"
        after_dir.mkdir(parents=True, exist_ok=True)

        if step < self.stream.config.total_steps:
            batch = self.stream.get_batch(step)
            fork_batch_id = batch.batch_id
            mean_loss, token_losses = fork_model.train_batch(batch)
            offset = fork_ledgers.offset
            opus_counts = Counter(d["decision"] for d in batch.opus_decisions)
            doc_ids = sorted({d for s in batch.sequences for d in s.doc_ids})
            lanes = sorted({ln for s in batch.sequences for ln in s.lanes})
            fork_ledgers.append_consumption(
                ConsumptionRecord(
                    step=batch.step,
                    batch_id=batch.batch_id,
                    batch_hash=batch.batch_hash,
                    stage=batch.stage,
                    doc_ids=doc_ids,
                    lanes=lanes,
                    useful_tokens=sum(s.useful_tokens for s in batch.sequences),
                    pad_tokens=sum(s.pad_tokens for s in batch.sequences),
                    token_span_hashes=[s.sequence_hash for s in batch.sequences],
                    opus_summary=dict(opus_counts),
                    ledger_offset=offset,
                )
            )
            fork_ledgers.append_learning(
                LearningRecord(
                    step=batch.step,
                    batch_id=batch.batch_id,
                    batch_hash=batch.batch_hash,
                    mean_loss=mean_loss,
                    token_losses=token_losses[:256],
                    loss_linked_docs=doc_ids,
                    perplexity=float(math.exp(min(mean_loss, 20.0))),
                    ledger_offset=offset,
                )
            )
            fork_ledgers.save(after_dir / "ledgers")
            fork_model.save(after_dir / "model.npz", extra={"branch": branch, "parent": source_ckpt})
            meta2 = CheckpointMeta(
                step=step + 1,
                ledger_offset=fork_ledgers.offset,
                batch_id_next="",
                batch_hash_next=None,
                branch=branch,
                parent_checkpoint=source_ckpt,
                tokenizer_hash=self.tokenizer_hash,
                model_path=str(after_dir / "model.npz"),
            )
            (after_dir / "meta.json").write_text(
                json.dumps(asdict(meta2), indent=2, sort_keys=True), encoding="utf-8"
            )

        self.ledgers = main_ledgers
        return {
            "fork_ckpt": dest,
            "branch": branch,
            "from_step": step,
            "fork_continued_batch": fork_batch_id,
            "main_ledger_len": len(main_ledgers.consumption),
            "after_ckpt": str(after_dir.name),
        }

    def finalize_perf(self) -> dict[str, Any]:
        elapsed = max(self.perf.get("elapsed_sec", 1e-9), 1e-9)
        useful = self.perf.get("useful_tokens", 0)
        util = self.stream.packing_report(
            [self.stream.get_batch(i) for i in range(self.stream.config.total_steps)]
        )
        report = {
            "useful_tokens_total": useful,
            "elapsed_sec": elapsed,
            "useful_loss_tokens_per_sec": useful / elapsed,
            "packing_utilization": util["utilization"],
            "pad_fraction": util["pad_fraction"],
            "packing_detail": util,
        }
        self.perf = report
        (self.artifacts / "performance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        self.log(
            f"[EVENT] performance_measured useful_tps={report['useful_loss_tokens_per_sec']:.1f} util={util['utilization']:.3f}"
        )
        return report
