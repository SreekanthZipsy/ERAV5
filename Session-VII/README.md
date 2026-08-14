# Kronecker Embeddings V2 — Multimodal Atom×Coordinate Factorization

Extends [Kronecker Embeddings](https://arxiv.org/abs/2605.29459) (Shravan, 2026) from text-only byte×position structure to **text, image, and audio** with one shared rule.

---

## Problem

V1 replaces a huge `|V|×d` embedding table with a deterministic **byte ⊗ byte-position** codec plus a thin learned projection. That works for language tokens.

The open question for V2:

> **What is the natural extension of Kronecker so the same idea can represent images and audio — not only text?**

Constraints we refuse to break:

1. Keep a **fixed, structured codec** (not a new giant lookup table per modality).
2. Keep a **shared transformer interface**: every modality must emit `e ∈ R^{d_model}`.
3. Prefer **locality**: similar surface structure → similar κ (V1’s inductive bias).
4. Prove it with **code + measurements**, not slides.

Images need preprocessing into patches; audio into frames. The hard part is not patching — it is showing that after patching, Kronecker still applies.

---

## Solution

**One ontology for all three modalities:** every unit is a set `S` of **(atom, coordinate)** events.

```
κ(S) = |S|^{-1/2}  Σ_{(a,c) ∈ S}  φ(a) ⊗ ψ(c)
e     = κ W_proj     ∈ R^{d_model}
```

| Modality | Unit | Atom `a` | Coordinate `c` | Codec |
|----------|------|----------|----------------|-------|
| **Text** | BPE/char token | UTF-8 byte | byte index in token | `byte ⊗ pos` |
| **Image** | patch | median-relative rank | `(row, col)` in patch | `atom ⊗ row ⊗ col` |
| **Audio** | frame | μ-law amplitude bucket | time in frame | `q ⊗ time` |

Why this is the natural extension:

- V1 already said “token = bag of (byte, position)”.
- An image patch is a bag of (structure-atom, spatial coord).
- An audio frame is a bag of (quantized sample, time).
- Images use a **triple** product because space is 2D — still Kronecker algebra (`A⊗B⊗C`), not a different paradigm.

The transformer body is unchanged. Only the front-end codec differs by modality; all paths share `d_model`.

---

## How we prove it

Evidence is **generated** by the implementation (`python run_demo.py` → `artifacts/results.json`). Nothing in the verdict table is hardcoded.

### Proof 1 — Locality (geometry of κ)

If the factorization is right, structure-preserving changes keep high cosine; structure-destroying ones do not.

| Probe | Expectation |
|-------|-------------|
| Image: brightness affine | high cosine to original |
| Image: pixel shuffle / other shape | low cosine |
| Audio: additive noise | high cosine |
| Audio: different pitch | low cosine |
| Text: typo / related spellings | higher overlap than unrelated strings |

### Proof 2 — Trainability (same tiny transformer)

A 2-layer transformer trains on Kronecker inputs for three tasks, vs a learned-table / hash-table baseline:

| Task | Kronecker input | Baseline |
|------|-----------------|----------|
| Character LM | 1-byte tokens → TextCodec | `nn.Embedding` |
| Shape classification | patches → ImageCodec | hash → table |
| Tone classification | frames → AudioCodec | hash → table |

**Pass criterion:** Kronecker front-ends train successfully (loss falls / accuracy rises) and stay competitive with the baseline at this scale.

### Proof 3 — Parameter structure

At frontier-style text settings (`|V|=131072`, `d=4096`), input cost is `D×d` (codec width × model width), not `|V|×d`. Image/audio never needed a huge discrete vocab table; they get the same structured projection pattern as text.

---

## One command

```bash
cd Session-VII
pip install -r requirements.txt
python run_demo.py
python -m pytest tests/ -q
```

Static report (problem, solution, live numbers, plots):

```bash
cd report && python -m http.server 8080
# → http://localhost:8080
```

---

## Layout

```
Session-VII/
  kronecker_v2/     # codecs, data, tiny models, experiments
  run_demo.py       # end-to-end proof + report generator
  artifacts/        # results.json (machine evidence)
  report/           # index.html + plots/
  tests/
```

---

## Honest limits

- Patch / frame / quantization knobs are preprocessing choices.
- Triple Kronecker grows with `levels × H × W` — keep patches small or learn an atom codebook at scale.
- This repo proves the **idea at small scale**. Frontier claims remain extrapolations from V1’s controlled study plus these multimodal locality and trainability probes.
