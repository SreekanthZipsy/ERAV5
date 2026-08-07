"""Generate evidence.json / evidence.md from real run artifacts (never hardcoded)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_evidence(artifacts: Path, checks: dict[str, Any]) -> tuple[dict, str]:
    artifacts = Path(artifacts)
    rows = []

    def add(req: str, key: str, evidence_path: str) -> None:
        ok = bool(checks.get(key, False))
        rows.append(
            {
                "requirement": req,
                "result": "PASS" if ok else "FAIL",
                "evidence": evidence_path,
                "detail": checks.get(f"{key}_detail"),
            }
        )

    add("Tokenizer integrity", "tokenizer_integrity", "manifests/index.json")
    add("Evaluation firewall", "eval_firewall", "ledgers/firewall_events.json")
    add("Packing correctness", "packing", "performance.json")
    add("Mixture compliance", "mixture", "ledgers/mixture_report.json")
    add("OPUS audit trail", "opus", "ledgers/opus_decisions.jsonl")
    add("Crash recovery", "resume", "ledgers/resume_proof.json")
    add("Replay", "replay", "ledgers/replay_proof.json")
    add("Learning trace", "learning", "ledgers/learning.jsonl")
    add("Throughput", "throughput", "performance.json")
    add("Fork", "fork", "checkpoints/")

    evidence = {
        "schema": "era5-tdes-evidence-v1",
        "all_pass": all(r["result"] == "PASS" for r in rows),
        "checks": rows,
        "raw": {k: v for k, v in checks.items() if not k.endswith("_detail")},
    }

    md_lines = [
        "# Evidence Summary",
        "",
        "| Requirement | Result | Evidence |",
        "|---|---|---|",
    ]
    for r in rows:
        md_lines.append(f"| {r['requirement']} | {r['result']} | {r['evidence']} |")
    md_lines.append("")
    md_lines.append(f"**Overall:** {'PASS' if evidence['all_pass'] else 'FAIL'}")
    md = "\n".join(md_lines) + "\n"

    (artifacts / "evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    (artifacts / "evidence.md").write_text(md, encoding="utf-8")
    return evidence, md
