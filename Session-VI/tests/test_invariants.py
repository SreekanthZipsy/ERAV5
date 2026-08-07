"""Invariant tests for the Training Data Execution System."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tdes.corpus import build_demo_corpus
from tdes.firewall import EvalFirewall
from tdes.mixture import PROTECTED_FLOORS, enforce_floors
from tdes.packing import pack_spans
from tdes.shards import tokenize_documents, write_shards
from tdes.stream import PACKING_BY_LANE, BatchStream, StreamConfig
from tdes.tokenizer import FrozenTokenizer, build_tokenizer_from_texts


@pytest.fixture
def tok_and_spans(tmp_path):
    docs = build_demo_corpus()
    tok = build_tokenizer_from_texts([d.text for d in docs])
    spans = tokenize_documents(docs, tok)
    return tok, spans, docs


def test_tokenizer_hash_stable(tok_and_spans, tmp_path):
    tok, _, _ = tok_and_spans
    p = tmp_path / "tok.json"
    h = tok.save(p)
    tok2 = FrozenTokenizer.load(p)
    assert tok2.hash == h


def test_eval_firewall_blocks(tok_and_spans):
    _, spans, _ = tok_and_spans
    fw = EvalFirewall()
    kept = fw.filter_train_spans(spans)
    assert all(s.split == "train" for s in kept)
    assert any(b.split == "eval" for b in fw.blocked)


def test_agentic_loss_mask_hides_observations(tok_and_spans):
    tok, spans, _ = tok_and_spans
    agent = next(s for s in spans if s.lane == "agentic")
    assert sum(agent.loss_mask) < len(agent.token_ids)
    assert 0 in agent.loss_mask


def test_packing_mask_alignment(tok_and_spans):
    tok, spans, _ = tok_and_spans
    train = [s for s in spans if s.split == "train" and s.lane == "web"]
    packed = pack_spans(train, tok, seq_len=32, policy="greedy_concat")
    for p in packed:
        assert len(p.input_ids) == len(p.loss_mask) == len(p.attention_mask) == len(p.position_ids) == 32
        # pad positions must not bear loss
        for i, tid in enumerate(p.input_ids):
            if tid == tok.spec.pad_id:
                assert p.loss_mask[i] == 0
                assert p.attention_mask[i] == 0


def test_protected_floors():
    w = enforce_floors({"web": 0.9, "indic": 0.01, "code": 0.05, "agentic": 0.01})
    assert w["indic"] >= PROTECTED_FLOORS["indic"] - 1e-9
    assert abs(sum(w.values()) - 1.0) < 1e-6


def test_shard_immutability(tok_and_spans, tmp_path):
    tok, spans, _ = tok_and_spans
    mans = write_shards(spans, tok, tmp_path, PACKING_BY_LANE)
    assert mans
    # rewriting same content ok; different content should fail
    write_shards(spans, tok, tmp_path, PACKING_BY_LANE)


def test_stream_deterministic(tok_and_spans):
    tok, spans, _ = tok_and_spans
    fw = EvalFirewall()
    train = fw.filter_train_spans(spans)
    s1 = BatchStream(train, tok, EvalFirewall(), StreamConfig(total_steps=5, seed=7))
    s2 = BatchStream(train, tok, EvalFirewall(), StreamConfig(total_steps=5, seed=7))
    b1 = s1.materialize_all()
    b2 = s2.materialize_all()
    assert [b.batch_hash for b in b1] == [b.batch_hash for b in b2]


def test_demo_artifacts_if_present():
    art = ROOT / "submission_artifacts"
    if not art.exists():
        pytest.skip("run demo first")
    evidence = json.loads((art / "evidence.json").read_text())
    assert evidence["all_pass"]
    log = (art / "run.log").read_text()
    for marker in (
        "[PASS] tokenizer_hash_verified",
        "[PASS] eval_shard_blocked",
        "[PASS] checkpoint_saved",
        "[PASS] resume_next_batch_matched",
        "[PASS] replay_hash_matched",
    ):
        assert marker in log
