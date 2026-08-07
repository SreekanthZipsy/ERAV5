# ERA5 Session-VI — Training Data Execution System (TDES)

Small but complete path from documents → audited, resumable training.

## One command

```bash
cd Session-VI
python run_demo.py
```

Then:

```bash
python -m pytest tests/ -q
```

Artifacts land in `submission_artifacts/`.

## Architecture

```
documents.jsonl
    → FrozenTokenizer (content-hashed)
    → immutable tokenized shards + manifests
    → EvalFirewall (blocks eval/validation)
    → Mixture schedule (curriculum stages + protected floors)
    → OPUS accept/reject/defer/floor_override
    → Packing (greedy / best_fit / structure_preserving) + masks
    → TinyLM train step
    → consumption + learning ledgers
    → checkpoints (model + optimizer moments + ledger offset + next batch id/hash)
    → crash → resume → replay → fork → evidence bundle
```

### Design decisions

| Topic | Choice | Why |
|-------|--------|-----|
| Scale | Tiny corpus + numpy LM | Prove correctness, not FLOPs |
| Shards | Content-addressed, refuse mutate | Immutability / LakeFS-style identity |
| Tokenizer | Frozen file + hash check on load | Same recipe as Megatron “don’t retokenize mid-run” |
| Packing | Lane-specific policies | Class rule: web/code concat OK; agentic structure-preserving |
| Loss masks | Prompt/OBS=0, answer/calls=1 | Agentic traces must not train on tool logs |
| Floors | Indic/code/agentic minima | OPUS EN-heavy proxies would otherwise starve them |
| OPUS | Cheap quality×novelty proxy | Same control flow as gradient OPUS; auditable decisions |
| Resume | Checkpoint stores **next** batch id+hash + ledger offset | Mosaic-style mid-epoch resume without skip/repeat |
| Replay | Rematerialize planned stream; compare batch hashes | Reconstructibility |
| Fork | Copy checkpoint tree; train on branch without touching main ledger | Experiment branches |

### Packing & masks

Each packed sequence carries aligned:

- `input_ids`
- `loss_mask` (1 = CE target)
- `attention_mask` (0 on pad)
- `position_ids` (restart after `<eos>` when concatenating)

### Crash / resume proof

Demo trains steps `0..5`, saves `ckpt_pre_crash_step6`, raises `SIMULATED_CRASH`, resumes, and asserts the next batch id/hash match the checkpoint. Consumption ledger steps must equal `0..11` exactly once.

### Replay proof

Steps `[0, crash)` are rebuilt from the deterministic stream and compared to archived `batch_hash` values.

## Artifact layout

```
submission_artifacts/
  run.log
  evidence.json
  evidence.md
  manifests/          # tokenizer, shards, index
  ledgers/            # consumption, learning, opus, firewall, proofs
  checkpoints/        # pre-crash, periodic, fork
  performance.json
```

## Requirements

- Python 3.10+
- `numpy`
- `pytest` (tests only)
