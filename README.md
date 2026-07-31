# Session V — Mixture & Curriculum Spec (Bhārat-V5)

**Verdict.** Mixture is DNA, not a spreadsheet. This plan locks a **4.0T-token** pretrain budget (class range 2.4–4T; proportions scale to 8T unchanged), composed **backward from the win set** (coding + agentic + controllable reasoning + Indic), with scarce lanes held small early and concentrated in mid-train / anneal. Every number below is a **hypothesis** until a 1B/3B proxy confirms or kills it.

---

## 1. Objective (what the mixture must buy)

The model must: (1) code and debug in real repos/terminals, (2) plan multi-step tool use, recover on failure, hold growing history, (3) reason at controllable depth (instant → ultra), (4) stay Indic-capable under English flood. Web stays large because **common sense lives on the web** — code without world knowledge writes crash-free nonsense.

---

## 2. Capability budget (main pretrain, % of 4.0T)

| Slot | Share | Tokens @ 4T | Primary benchmarks | Inventory fill |
|------|------:|------------:|--------------------|----------------|
| **General web** | **30%** | 1.20T | MMLU / MMLU-Pro, world knowledge | FineWeb / DCLM-class filtered crawl; books; high-quality news |
| **Code** | **18%** | 720B | LiveCodeBench, Codeforces, SWE-bench / Live-Pro | The Stack v2 (license-clean), issue→PR pairs, OSS-Instruct-class synthetic (cap below) |
| **Indic** | **18%** | 720B | IndicMMLU-Pro, IndicGenBench, FLORES-IN, MEGA | Sangraha + Setu/CulturaX + parallel + synthetic — **tiered in §3** |
| **STEM / math** | **10%** | 400B | AIME, MATH, GPQA | arXiv, ProofWiki, OpenR1-Math (cleaned Session IV), textbooks |
| **Reasoning traces** | **8%** | 320B | LiveBench, AIME (pass@k + effort tags) | OpenR1 / contest CoT; distilled effort-tagged traces (§6) |
| **Agentic / tools** | **6%** | 240B | Terminal-Bench, τ²-Bench, BFCL, WebArena / BrowserComp | ToolBench, API multi-tool chains, sandbox SWE/terminal trajectories — **supply-starved** |
| **Long context** | **5%** | 200B | Long-horizon / needle / multi-doc / repo-scale | Packed books, multi-file repos, long agent trajectories at full seq length |
| **India context** | **5%** | 200B | SANSKRITI / Indica / held-out UPSC–legal–gov | Gazettes, judgments, RBI/SEBI, PIB, NCERT, regional news |

**Sum = 100%.** Scarce lanes (agentic, verified Indic, long reasoning) are **intentionally small in the bulk run** and **upweighted in mid-train + anneal** (§5), matching the class rule: don’t dump PhD traces on a model that still says “abad.”

### Why these shares (reviewer pushback)

- **Web 30% not 50%:** We need coding/agent headroom, but cutting web below ~25% historically tanks MMLU-style GK. 30% is the floor that still buys common sense.
- **Code 18%:** Stack v2 alone can supply hundreds of billions of tokens — this share is **inventory-backed**, not wishful. Synthetic code ≤25% of the code slot.
- **Indic 18% not 30%:** Session-III’s 30% assumed heroic collection. Class reality: multi-trillion native Indic does not exist without heavy repeat. 18% with a **hard always-on floor** (§4) beats a fake 30% of mostly English-translated sludge.
- **Agentic 6%:** ToolBench-scale public data is **tens of millions of tokens**, not hundreds of billions. Hitting 6% **requires** (a) cohort-collected Cursor/Claude-Code trajectories, (b) sandbox rollouts, (c) **≤3× repeat** of gold traces, (d) mid-train concentration. Claiming 15%+ from public APIs alone is dishonest accounting.
- **Long context 5%:** Length ≠ volume. This slot is about **sequence length stages** (4k→64k+), not dumping short docs. Most “long” supply must be built (repo packs, concatenated verified docs, full trajectories).

---

## 3. Indic slot — verified / unverified / translated / synthetic

Of the **18% Indic budget (720B)**:

| Tier | % of Indic | Tokens | What counts | Rule |
|------|-----------:|-------:|-------------|------|
| **Verified** | **35%** | 252B | Sangraha Verified, curated Wikipedia/news, human-audited books/gov OCR | Highest mix rank; **anneal-eligible** |
| **Unverified** | **25%** | 180B | Setu / CulturaX web after LID + quality + dedup | Allowed in main; **blocked from anneal** |
| **Translated** | **22%** | 158B | IndicTrans2 / NLLB EN→Indic of STEM, code comments, India-policy EN | Must keep parallel EN for consistency checks |
| **Synthetic** | **18%** | 130B | LLM-generated Indic (chat, explanations, Hinglish) | Cap; quality-filter vs Verified KenLM |

**Hard caps.** `(translated + synthetic) ≤ 40%` of Indic. No language may be >50% synthetic. Hinglish/Tanglish = explicit **2% of total budget** carved from Indic unverified + synthetic (not accidental noise).

**Language tiers (within Indic tokens):** T1 (hi, bn, te, mr, ta, ur, gu, kn, ml, or, pa) **70%** · T2 **20%** · T3 **10%** (upsample + tokenizer oversample). Sanskrit/Urdu literary: mid-train only, not nursery.

**Supply honesty.** Verified native supply is the bottleneck. Gap to 252B verified → **repeat verified ≤2×** before filling with translated; never fill the verified bucket with synthetic and relabel.

---

## 4. Protected always-on floor (selector must not cross)

Online selection (Opus-style gradient affinity) will **starve Indic and agentic**: proxy benches are EN-heavy; agent traces look like “logs” in the first 512 tokens. Therefore the selector may reweight freely **except**:

| Always-on lane | Floor (every selected batch) | Rationale |
|----------------|-----------------------------:|-----------|
| **Indic (all tiers)** | **≥12%** | Prevents English wipeout of the differentiator |
| **of which Verified Indic** | **≥4%** absolute (≥⅓ of Indic floor) | Stops “Indic” becoming translation mush |
| **Agentic / tools** | **≥3%** | First-512 Opus truncation would drop long traces |
| **Code** | **≥10%** | Protects primary win condition under GK-heavy proxies |

If the selector’s keep-fraction would breach a floor, **pad from the protected pools** before accepting the batch. Floors are not targets — targets are §2; floors are anti-erase.

---

## 5. Anneal reserve (cooldown)

**Hold back 2.0% of total budget = 80B tokens** never seen in the main run. Released only in the final **~2% of training steps** (cooldown / mid-train lean), LR low, same CE loss.

**Reserve composition (of the 80B):**

| Content | Share of reserve | Notes |
|---------|-----------------:|-------|
| Verified Indic gold | 30% | PhD-clean docs only |
| Long reasoning (high/ultra) | 25% | Effort-tagged, answer-checked |
| Agentic recovery trajectories | 25% | Loss on assistant/tool-*calls* only; **mask tool observations** |
| STEM proofs / textbooks | 15% | Step-valid proofs preferred |
| Long-context packs (≥32k) | 5% | Full sequences; no chop-to-4k |

**Do not anneal on:** raw web, unverified Indic, synthetic chat, ToolBench-short stubs.

---

## 6. Curriculum phases (when, not only how much)

Smooth band overlap ≥15% of tokens between adjacent phases (class: sharp mixture shifts spike grad norm; target stable ~0.2 later in run).

| Phase | Progress | Mixture emphasis | Seq length |
|-------|----------|------------------|------------|
| **B0 Foundation** | 0–35% | Web 40 · Indic 15 · Code 12 · STEM 10 · India 5 · Reasoning 5 · Agent 3 · Long 2 · *(rest buffer)* | 4k |
| **B1 Skills** | 35–65% | Web 28 · Code 20 · Indic 18 · STEM 12 · Reasoning 8 · Agent 5 · Long 4 · India 5 | 4k→8k |
| **B2 Lean / college** | 65–90% | Web 22 · Code 22 · Indic 16 · Reasoning 12 · Agent 10 · STEM 8 · Long 6 · India 4 | 8k→16k |
| **B3 Pre-anneal** | 90–98% | Web 18 · Code 20 · Agent 14 · Reasoning 14 · Long 10 · Indic 14 · STEM 6 · India 4 | 16k→32k |
| **B4 Anneal** | 98–100% | **Reserve only** (§5) | 32k→64k (separate same-length batches) |

Phase row % are **targets inside the phase**, renormalized; lifetime average still tracks §2. Long-context rule: **one sequence length per batch**; never mix 4k and 16k in one batch; pad is wasted compute — pack or drop.

**Agentic loss mask (all phases that use traces):** train on assistant tokens + tool-*call* tokens; **do not** train on tool observation / shell logs / compiler dumps (context only). Same for SWE: issue text and test logs are observation; patch/tool calls are loss.

---

## 7. Difficulty bands (with one concrete example each)

Ladder inside STEM + reasoning + code. Overlap 10–20% between adjacent bands when transitioning.

| Band | When introduced | Example |
|------|-----------------|---------|
| **Nursery** | B0 | “What is 43 ÷ 17?” → short mental math (~2.5), no olympiad setup |
| **Grade** | B0→B1 | “How many integers from 1 to 1000 are divisible by 3 or 5?” → inclusion–exclusion, answer 467 |
| **Undergrad** | B1 | Prove / derive a standard textbook result (e.g. plate-tectonics explanation with causal chain; or implement BFS with tests) |
| **Grad** | B2 | Multi-file bug localization in a small repo; or contest problem needing 2–3 lemmas |
| **Research / PhD** | B3–B4 | Frontier-style math or long agent investigation (search → fail → recover → synthesize) — **anneal only if answer-verified** |

---

## 8. Reasoning-length bands (controllable depth)

Effort is a **trained tag**, not “truncate at N tokens.” Each example carries an effort label; the model must match depth *and* keep answer quality.

| Effort | Typical useful think length | Concrete example |
|--------|----------------------------:|------------------|
| **Instant** | ≤50 tokens | “Capital of India?” → `New Delhi` |
| **Low** | ~100–400 | “43 ÷ 17 ≈ ?” → brief estimate, no formal proof |
| **Medium** | ~400–2k | Inclusion–exclusion count (1..1000, ÷3 or ÷5) with written steps |
| **High** | ~2k–8k | Multi-hop agent plan: find researchers in a field, filter, draft outreach — with tool calls |
| **Ultra** | ≥8k (often hours-of-wall-clock distilled) | Hard contest / research math or long-horizon app build with recoveries |

**Supply note.** Instant/low exist; high/ultra **must be generated** (distill from strong models at fixed effort, or collect cohort traces). Budget: within the 8% reasoning slot — Instant 10% · Low 20% · Medium 35% · High 25% · Ultra 10% (ultra mostly anneal).

---

## 9. Agentic / reasoning / long-context — explicit dataset map

| Slot | Datasets that fill it | What we still lack |
|------|----------------------|--------------------|
| **Agentic** | ToolBench (multi-API); τ²-style retail/airline policies; BFCL function schemas; Terminal-Bench tasks; WebArena/BrowserComp-style browse traces; **cohort Cursor/Claude-Code trajectories** | Volume: public tool data ≪ 240B. Plan: generate sandbox rollouts + ≤3× gold repeat + mid-train upweight |
| **Reasoning** | OpenR1-Math-220k (Session IV cleaned ~7.8M tok slice — **scale cleaning**); contest CoT; effort-tagged distillates | Ultra traces; Indic-language CoT |
| **Long context** | Book packs; multi-file Stack repos packed to target L; full agent trajectories at L; legal/gov long docs (India) | True ≥64k packs; must train at target L (half-length does not prove full L) |

---

## 10. Proxy experiments (hypothesis → cheap falsification)

**No full-scale trust without this.** Run before locking the 4T recipe.

### Proxy A — 1B dense, ~20–40B tokens

| Arm | Mixture tweak | Confirm if… | Kill if… |
|-----|---------------|-------------|----------|
| A0 | Flat §2 proportions, no curriculum | Baseline loss & eval | — |
| A1 | **−8% web, +8% code** | LiveCodeBench-mini ↑ ≥3 pts vs A0, MMLU-mini drop ≤1.5 | Code↑ but MMLU collapse >3 pts → web too low |
| A2 | Indic floor **off** vs **on** (§4) | With floor: IndicEval-mini drop ≤1 across training; without: ≥5 pt bleed | Floor on but no Indic lift → data quality, not % |
| A3 | Agentic 2% vs 6% (rest from web) | BFCL-lite / simple tool accuracy ↑ with 6% | No lift → format/mask bug, not share |

**Primary metric:** Pareto of `{LiveCodeBench-mini, BFCL-lite, Indic bit of MMMLU/IndicEval-mini, MMLU-mini}`. Accept mixture iff code+agent+Indic all non-regress vs A0 within thresholds above.

### Proxy B — 3B dense, ~60–100B tokens

| Arm | Question | Metric |
|-----|----------|--------|
| B1 | Does B0→B2 curriculum beat flat mix at same tokens? | Same Pareto; also **grad norm stability** (fewer spikes at phase boundaries) |
| B2 | Anneal 2% reserve vs spend reserve in main | Final 5% steps: reasoning-high + tool recovery ↑ without MMLU loss |
| B3 | Long-ctx: train 8k-only vs staged 4k→8k→16k | Needle / multi-doc at 16k; staged must win |

**Decision rule.** Promote a share to full scale only if **both** 1B and 3B agree on sign of the effect. If they disagree, run a third 1B seed before believing either.

### Optional executed stub (this repo)

Session IV already cleaned an OpenR1 slice (**45.0M → 7.84M tokens**). Next cleaning priority for this mixture: **expand OpenR1 + start ToolBench/terminal trace clean** into the starved agentic and high-reasoning bands — not more FineWeb.

---

## 11. Cleaning priority (cumulative target → starved slots)

Order by mixture gap, not by what’s easy:

1. **Agentic trajectories** — mask observations; keep call/patch tokens; PII; decontaminate Terminal-Bench / SWE test splits  
2. **Verified Indic** — Sangraha Verified + gov/book OCR; KenLM vs verified; no synthetic leakage into verified  
3. **High/ultra reasoning** — OpenR1-scale; effort tags; answer verify  
4. **Long packs** — repo/book concatenation at target L; dedup across packs  
5. **Code license + decontam** — Stack permissive only; strip benchmark solutions  

Web quality continues in background; it is not the bottleneck for *this* plan.

---

## 12. One-page numbers card (for defense)

```
Budget:        4.0T pretrain (+ 80B anneal reserve = 2% held back)
Lanes:         Web 30 | Code 18 | Indic 18 | STEM 10 | Reason 8 | Agent 6 | Long 5 | India 5
Indic tiers:   Verified 35 | Unverified 25 | Translated 22 | Synthetic 18  (trans+synth ≤40%)
Always-on:     Indic ≥12 (Verified ≥4) | Agent ≥3 | Code ≥10
Anneal:        80B = Verified Indic 30 | Long reason 25 | Agent recover 25 | STEM 15 | Longctx 5
Seq schedule:  4k → 8k → 16k → 32k → 64k (homogeneous batches)
Proxy gate:    1B + 3B Pareto on LCB-mini / BFCL-lite / IndicEval-mini / MMLU-mini
Starved:       Agentic volume, Verified Indic, Ultra traces  → clean these next
```

---

## 13. What this plan refuses

- A single headline “Indic 30%” without tier split  
- Agentic 15%+ funded only by ToolBench token counts  
- Anneal on raw crawl  
- Letting Opus drop Indic/agent to zero  
- Claiming 1M context without training near that length  
- Trusting mixture percentages that never survived a 1B/3B proxy  

*A data decision is a hypothesis until a cheap experiment has tested it.*
