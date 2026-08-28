# The Attention Bill — Session VIII

Interactive chronological map of attention mechanisms: each one as an answer to a problem on the **compute / memory / length** bill that scaled dot-product attention created.

- **Live app:** deploy this folder to Netlify (publish directory = this folder).
- **Local preview:** open `index.html`, or `python3 -m http.server` from this directory.

## Minimum coverage (all present)

| Required | Card id | Timeline date | Primary source |
|----------|---------|---------------|----------------|
| Standard attention | `scaled-dot-product` | 2017-06-12 | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| Sinusoidal PE | `sinusoidal` | 2017-06-12 | same paper |
| Absolute learned PE | `learned-absolute` | 2017-06-12 | Transformer §3.5; GPT popularized 2018-06-11 |
| Sparse & top-k | `sparse-topk` | 2019-04-23 | [arXiv:1904.10509](https://arxiv.org/abs/1904.10509); Routing Transformer [2003.05997](https://arxiv.org/abs/2003.05997) |
| MQA | `mqa` | 2019-11-06 | [arXiv:1911.02150](https://arxiv.org/abs/1911.02150) |
| Sliding window | `sliding-window` | 2020-04-10 | [arXiv:2004.05150](https://arxiv.org/abs/2004.05150) |
| Linear attention | `linear-attention` | 2020-06-29 | [arXiv:2006.16236](https://arxiv.org/abs/2006.16236) |
| Delta rule | `delta-rule` | 2021-02-22 | [arXiv:2102.11174](https://arxiv.org/abs/2102.11174) |
| RoPE | `rope` | 2021-04-20 | [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) |
| ALiBi | `alibi` | 2021-08-27 | [arXiv:2108.12409](https://arxiv.org/abs/2108.12409) |
| GQA | `gqa` | 2023-05-22 | [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) |
| NTK-aware scaling | `ntk-aware` | **2023-06-30** | u/bloc97 r/LocalLLaMA; HF TGI [#512](https://github.com/huggingface/text-generation-inference/issues/512) |
| YaRN | `yarn` | 2023-08-31 | [arXiv:2309.00071](https://arxiv.org/abs/2309.00071) |
| Attention sinks | `attention-sinks` | 2023-09-29 | [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) |
| MLA | `mla` | 2024-05-07 | [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) |
| Gated DeltaNet | `gated-deltanet` | 2024-12-09 | [arXiv:2412.06464](https://arxiv.org/abs/2412.06464) |
| DeepSeek compressed sparse | `deepseek-nsa` | 2025-02-16 | [arXiv:2502.11089](https://arxiv.org/abs/2502.11089) (NSA) |
| DroPE | `drope` | 2025-12-13 | [arXiv:2512.12167](https://arxiv.org/abs/2512.12167) |

## Bonus (same standard)

| Mechanism | Date | Source |
|-----------|------|--------|
| FlashAttention | 2022-05-27 | [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) |
| DeepSeek DSA (V3.2-Exp) | 2025-09-29 | [DeepSeek announcement](https://api-docs.deepseek.com/news/news250929) |

## Date methodology (re-verified 2026-08-28)

1. **Default:** first arXiv `<published>` via `export.arxiv.org/api/query?id_list=…`.
2. **Same calendar day:** story order (attention → sinusoidal primary → learned alternative).
3. **Learned absolute PE:** first *appearance* is Transformer 2017-06-12 (§3.5), not GPT. GPT (2018-06-11) is noted as LM popularization.
4. **NTK-aware:** corrected from an earlier mid-June guess → **2023-06-30** (bloc97 post; same-day HF TGI issue). YaRN is the citable write-up.
5. **YaRN:** arXiv id `2309.00071` but `<published>` is **2023-08-31**.
6. **Sliding window:** Longformer = research launch; Mistral = popularization note only.
7. **DeepSeek stack (do not conflate):**
   - **MLA** (2024-05-07) — compressed KV latents
   - **NSA** (2025-02-16) — research compressed + sparse hierarchical attention
   - **DSA** (2025-09-29) — production top-k sparse on MLA (bonus card)
8. **DroPE:** Sakana AI / arXiv:2512.12167, 2025-12-13.

### Corrections made in this pass

| Item | Was | Now | Why |
|------|-----|-----|-----|
| NTK-aware date | ~2023-06-14 | **2023-06-30** | bloc97 / TGI#512 primary public day |
| Learned absolute date | 2018-06-11 (GPT) | **2017-06-12** | First appearance in Transformer §3.5 |
| Sparse & top-k | Bonus-only Sparse Transformer | **Required** class card | Assignment minimum |
| DeepSeek compressed sparse | Vague “NSA/DSA” blur | **NSA 2025-02-16** + bonus DSA | Separate compression (MLA) from compressed-sparse (NSA) |

## Question 2 — what the timeline shows

As a **list**, variants look like feature checkboxes. As a **timeline**, they are a mood swing:

1. **Exactness (2017)** — pay n² for softmax over every pair; bolt on position.
2. **Cut the matrix (2019–20)** — sparse/top-k, MQA, windows, linear fixed state.
3. **Buy length (2021–23)** — RoPE/ALiBi, then NTK/YaRN/sinks; FlashAttention as plumbing.
4. **Memory again (2024–25)** — MLA compression, gated delta, NSA/DSA sparse; DroPE treats PE as a scaffold.

That oscillation — **exactness → memory → length → memory** — is invisible in a family-grouped syllabus. Full essay is on the page (`#what-the-timeline-shows`).

## Deploy (Netlify)

1. Publish directory: `Session-VIII` (or `.` if this folder is the repo root).
2. Build command: none / `exit 0`.

## Files

```
Session-VIII/
  index.html
  css/styles.css
  js/mechanisms.js   # dated content + REQUIRED_IDS checklist
  js/app.js
  netlify.toml
  README.md
  transcript.txt     # class reference
```
