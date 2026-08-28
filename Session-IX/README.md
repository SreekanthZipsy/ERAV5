# Session IX — Loss harness

One notebook + one runnable harness for **loss functions and output heads** (ERA Session IX). The point of the assignment is not a fancy model — it is to make the standard LM loss **correct and observable**, then add one extra head and watch what happens.

## How to run

```bash
cd Session-IX
pip install -r requirements.txt
python run_harness.py
# or open Session_IX_Loss_Harness.ipynb and run all cells
```

| File | Role |
|------|------|
| [`Session_IX_Loss_Harness.ipynb`](Session_IX_Loss_Harness.ipynb) | Assignment notebook |
| [`loss_harness.py`](loss_harness.py) | Tiny GPT, tokenizer, CE / packing / chunked CE / MTP |
| [`run_harness.py`](run_harness.py) | CLI that runs every check |
| [`artifacts/results.json`](artifacts/results.json) | Numbers from the last run |

**Sample model:** 2-layer char-level GPT (`n_embd=64`, ~110k params), as allowed in class (“Karpathy GPT-2 / nano / any small architecture”). Char tokens make the shift check readable as strings.

**Class spine (what we make correct):**

```python
hidden = model(tokens)                         # (B, T, C)
logits = output_head(hidden)                   # (B, T, V)
loss = cross_entropy(
    logits[:, :-1].reshape(-1, vocab_size),    # predict FROM positions 0..T-2
    tokens[:, 1:].reshape(-1),                 # targets AT positions 1..T-1
)
```

---

## Short write-up (numbers from `artifacts/results.json`)

### Part 1 — seven numbers

| # | Check | Number(s) | What it means |
|---|--------|-----------|----------------|
| 1 | **Shapes** | tokens `(1, 25)` → hidden `(1, 25, 64)` → logits `(1, 25, 45)` | Batch × length × channels × vocab. After shift: `(1, 24, 45)` vs `(1, 24)`. |
| 2 | **Shift (strings)** | e.g. `T → h → e → ␣ → c…` | Printed input beside target as **characters**, not ids. Target is always the next char — that is how you catch an off-by-one. |
| 3 | **Pad mask** | contributing tokens **40 → 26** (14 pads excluded) | Without `ignore_index`, CE would train on `<pad>` as if it were a real next token. Masking changes the count; that is the proof. |
| 4 | **Packed boundary** | loss before **3.85925**, after **3.85944**; targets **79 → 78** | Two docs in one sequence. Masking the seam drops the bogus pair (end of A → start of B). Count falls by 1; mean CE moves slightly because that one pair is removed from the average. |
| 5 | **Untrained PPL** | uniform **PPL = 45.00** (= V); model **PPL = 46.63** (PPL/V ≈ **1.04**) | By reading: `PPL = exp(CE)`. Uniform ⇒ `CE = ln V` ⇒ `PPL = V`. If this is wrong, stop — the harness is buggy (we once saw astronomical PPL from bad init; fixed with GPT-2-scale weights). |
| 6 | **Tied vs untied** | unique params tied **110,656** vs untied **113,536**; **saved = 2,880** | Tying `lm_head.weight = tok_emb.weight` saves exactly `V × C = 45 × 64 = 2880`. Same matrix, two roles. |
| 7 | **CE memory** | full peak **65,536,000** B; chunked **4,096,000** B; **ratio = 16.0×**; losses both **9.168707** | Hand-written chunked CE matches full CE math; peak activation for logits drops 16× on `(B=4, T=512, C=128, V=8000)` with chunk 128. Same bill, smaller peak RAM. |

### Part 2 — two losses (plus sum)

Second head predicts token **t+2** from the same hidden state: `logits_t2[:, :-2]` vs `tokens[:, 2:]`. We train `loss_sum = L_t1 + L_t2` (MTP as in class: one token into the trunk; extra heads are supervision only).

| Step | **L_t1** (next) | **L_t2** (t+2) | Sum |
|------|-----------------|----------------|-----|
| 0 | 3.813 | 3.810 | 7.622 |
| 40 | 2.163 | 2.385 | 4.548 |
| 100 | 1.360 | 1.520 | 2.881 |
| 199 (final) | **0.426** | **0.428** | **0.854** |

**What we see:** both start near `ln(V) ≈ 3.81`. Through mid-training **L_t1 stays below L_t2** — next-token is the denser, easier signal; from position `t`, token `t+2` has more branching uncertainty. Both fall because they share the trunk. On this tiny char corpus they nearly meet after heavy overfit; the mid-training gap is the part that matters.

---

## More explanation (why each check exists)

**Shapes.** If you cannot name `B`, `T`, `C`, `V`, you will mis-wire the reshape into CE. The assignment forces you to print them once, correctly.

**String shift.** Ids hide off-by-ones. Strings do not. Row `i` must show “predict FROM this char TO the next char.”

**Padding.** Batches are padded to a common length. Pads are not language. `ignore_index=-100` on pad **targets** is how you keep the mean CE honest. The contributing-token count must change.

**Document packing.** Packing saves compute by filling the context window, but invents a fake next-token at the seam. Mask that boundary or the model is punished for not predicting the start of an unrelated document.

**Perplexity.** This is the “read, don’t guess” check. Untrained ≈ uniform ≈ vocab size. Fail here and every later curve is meaningless.

**Tied head.** Output projection is often the largest matrix (`V × C`). Sharing it with the embedding table cuts unique params without changing the forward shape. Report both counts so the saving is visible.

**Chunked CE.** Class point: chunking does not change the math of mean NLL; it only changes peak memory. We measure both peaks and the ratio so that trade-off is a number, not a slogan.

**t+2 head.** Speculative / multi-token training adds heads that look further ahead. Training still advances one gold token at a time; inference is where speculation can skip forwards. Watching `L_t2` stay harder than `L_t1` is the expected signature.
