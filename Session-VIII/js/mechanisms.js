/**
 * Chronological attention mechanisms — dates cross-checked 2026-08-28.
 * Sort key: dateISO. Same-day ties: story order (mixer → PE primary → PE alternative).
 *
 * Date rule: first public appearance (arXiv <published>, or dated blog/report).
 * Every entry: problem, mechanism, buys, costs, when-to-pick.
 */
window.MECHANISMS = [
  {
    id: "scaled-dot-product",
    name: "Scaled Dot-Product Attention (Multi-Head)",
    short: "Standard attention",
    date: "2017-06-12",
    dateLabel: "Jun 2017",
    era: "exactness",
    family: "core",
    coveredInClass: true,
    paper: "Attention Is All You Need",
    authors: "Vaswani et al.",
    arxiv: "1706.03762",
    sourceNote:
      "Verified: arXiv API <published> 2017-06-12 for 1706.03762.",
    problem:
      "RNNs and CNNs made sequence transduction slow to train and hard to parallelize. Information had to travel hop-by-hop through time or stacked convolutions.",
    answer:
      "Compare every query to every key with a scaled dot product, softmax into a distribution, mix values. Multi-head runs several of these in parallel for different relationships.",
    formula: "Attention(Q,K,V) = softmax(QKᵀ / √dₖ) V",
    buys: [
      "Exact pairwise mixing — any token can look at any earlier token in one step",
      "Full parallelization over sequence length during training",
      "The quality ceiling every later trick is measured against"
    ],
    costs: [
      "O(n²) compute and activation memory for the score matrix",
      "KV cache grows with n × heads × dim at decode time",
      "Permutation-invariant alone — needs a position scheme"
    ],
    when:
      "Default when context is short-to-medium (~2–8K) and you can afford the bill.",
    pickFor: "2K–8K chatbot quality ceiling",
    skipFor: "1M-token agents without further tricks"
  },
  {
    id: "sinusoidal",
    name: "Sinusoidal Positional Encoding",
    short: "Sinusoidal PE",
    date: "2017-06-12",
    dateLabel: "Jun 2017",
    era: "exactness",
    family: "position",
    coveredInClass: true,
    paper: "Attention Is All You Need",
    authors: "Vaswani et al.",
    arxiv: "1706.03762",
    sourceNote:
      "Same paper, same day. Presented as the primary PE (§3.5).",
    problem:
      "Self-attention is permutation-invariant. Without position, 'bank' at token 2 and token 20 look identical to Q/K/V.",
    answer:
      "Add fixed sine/cosine waves of different frequencies to token embeddings so each absolute index has a unique geometric signature — no learned position table.",
    buys: [
      "Deterministic, parameter-free positions",
      "Defined for any length in form (practice still brittle past train L)",
      "Relative offsets are linearly recoverable from the encodings"
    ],
    costs: [
      "Absolute indices — the model must learn 'how far' indirectly",
      "Extrapolation beyond training length is brittle",
      "Injected at the input, so every submodule sees absolute position"
    ],
    when:
      "Fixed PE with no position parameters, lengths near the training regime. Largely superseded by relative schemes for modern LLMs.",
    pickFor: "Classic encoder/decoder MT near train length",
    skipFor: "Modern long-context decoder LLMs"
  },
  {
    id: "learned-absolute",
    name: "Absolute Learned Positional Embeddings",
    short: "Learned absolute PE",
    date: "2017-06-12",
    dateLabel: "Jun 2017",
    era: "exactness",
    family: "position",
    coveredInClass: true,
    paper: "Attention Is All You Need (§3.5 alternative); popularized by GPT",
    authors: "Vaswani et al.; Radford et al. (GPT)",
    arxiv: "1706.03762",
    sourceNote:
      "Appeared 2017-06-12 in Transformer §3.5 as a learned alternative ('nearly identical results'). Became the practical LM default with GPT (OpenAI report 2018-06-11) and BERT (arXiv:1810.04805, 2018-10-11). Timeline date = first appearance, not GPT popularization.",
    problem:
      "Sinusoids are one hand-designed inductive bias. You may want the model to reshape what each absolute index means during training.",
    answer:
      "Keep a trainable embedding vector per absolute index (0…L−1) and add it to token embeddings. The network learns what 'position 17' means.",
    buys: [
      "Simple and strong inside the trained length",
      "No hand-designed frequency schedule",
      "Matched sinusoids in the original Transformer ablations; dominated early GPT/BERT LMs"
    ],
    costs: [
      "Hard ceiling: indices never seen in training have no embedding",
      "Claiming 1M context means training with 1M-capable positions",
      "Absolute, not relative — moving a phrase changes its position IDs"
    ],
    when:
      "Short fixed context (BERT 512, early GPT). Wrong once you need length extrapolation.",
    pickFor: "Fixed 512–2K encoder/decoder LMs",
    skipFor: "Any model that must grow context after training"
  },
  {
    id: "sparse-topk",
    name: "Sparse & Top-k Attention",
    short: "Sparse / top-k",
    date: "2019-04-23",
    dateLabel: "Apr 2019",
    era: "compute",
    family: "sparse",
    coveredInClass: true,
    paper: "Generating Long Sequences with Sparse Transformers",
    authors: "Child, Gray, Radford, Sutskever",
    arxiv: "1904.10509",
    sourceNote:
      "Verified: arXiv 1904.10509 published 2019-04-23 (fixed/strided sparse patterns). Dynamic content-based / top-k lineage continues with Routing Transformer (arXiv:2003.05997, 2020-03-12). Class discussed the top-k idea; DeepSeek later productizes compressed+sparse selection.",
    problem:
      "Full attention's n² bill made very long sequences impractical even when you still wanted softmax exactness on the pairs you keep.",
    answer:
      "Do not score every pair. Use a sparse pattern (fixed/strided/window) or keep only the top-k keys per query (or a routed subset). Softmax runs only on the surviving edges.",
    buys: [
      "Sub-quadratic edges while keeping softmax on kept pairs",
      "Enables much longer sequences than dense MHA of the era",
      "Parent of sliding-window, sinks, and DeepSeek-style selectors"
    ],
    costs: [
      "Pattern or top-k choice is a prior — wrong subset drops the needle",
      "Irregular sparsity is hard on GPUs (load imbalance / gather)",
      "Too sparse → context collapse; too dense → you paid n² anyway"
    ],
    when:
      "Long sequences with locality or when a selector can be trained. Risky for adversarial needles if k is small.",
    pickFor: "Long structured sequences; trained top-k selectors",
    skipFor: "Safety-critical full-context audit over the whole past"
  },
  {
    id: "mqa",
    name: "Multi-Query Attention (MQA)",
    short: "MQA",
    date: "2019-11-06",
    dateLabel: "Nov 2019",
    era: "memory",
    family: "kv-sharing",
    coveredInClass: true,
    paper: "Fast Transformer Decoding: One Write-Head is All You Need",
    authors: "Shazeer",
    arxiv: "1911.02150",
    sourceNote: "Verified: arXiv 1911.02150 published 2019-11-06.",
    problem:
      "At decode time the bottleneck is often memory bandwidth reloading K and V — not FLOPs. Multi-head KV multiplies that cost by the head count.",
    answer:
      "Keep many query heads, but share a single K and a single V across all of them. The KV cache shrinks by roughly the head factor.",
    buys: [
      "Much smaller KV cache and faster incremental decoding",
      "Same softmax attention math — just shared K/V",
      "First clear 'pay the memory bill once' move"
    ],
    costs: [
      "Quality drop vs full multi-head on some tasks",
      "Less capacity for diverse key/value subspaces",
      "Training a separate MQA model (or converting) has a cost"
    ],
    when:
      "Inference-heavy serving where bandwidth dominates. Later softened by GQA.",
    pickFor: "High-QPS decode, memory-bound serving",
    skipFor: "Max-quality short-context research baselines"
  },
  {
    id: "sliding-window",
    name: "Sliding Window Attention",
    short: "Sliding window",
    date: "2020-04-10",
    dateLabel: "Apr 2020",
    era: "length",
    family: "sparse",
    coveredInClass: true,
    paper: "Longformer: The Long-Document Transformer",
    authors: "Beltagy, Peters, Cohan",
    arxiv: "2004.05150",
    sourceNote:
      "Verified: arXiv 2004.05150 published 2020-04-10. Mistral 7B popularized SWA in decoder LLMs (blog ~2023-09-27; arXiv:2310.06825 published 2023-10-10) — noted as popularization, not a second invention date.",
    problem:
      "Most tokens only need local context, but dense attention still materializes a global n×n score matrix.",
    answer:
      "Each token attends only to a fixed window of neighbors (optionally dilated / with global tokens). Stacked layers expand the receptive field like a CNN.",
    buys: [
      "O(n·w) attention for window width w",
      "Bounded KV if you only cache the window",
      "Practical path to long documents without full n²"
    ],
    costs: [
      "Pure windows forget the beginning of a long stream (sinks later fix this)",
      "Single layer cannot see far — needs depth",
      "Global facts outside the window are invisible unless you add globals/sinks"
    ],
    when:
      "Long documents with local structure; hybrid stacks. Alone, bad for 'needle in a 1M haystack'.",
    pickFor: "Long docs, local code context",
    skipFor: "Single-layer global recall over 1M tokens"
  },
  {
    id: "linear-attention",
    name: "Linear Attention (Softmax Removed)",
    short: "Linear attention",
    date: "2020-06-29",
    dateLabel: "Jun 2020",
    era: "memory",
    family: "linear",
    coveredInClass: true,
    paper: "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention",
    authors: "Katharopoulos et al.",
    arxiv: "2006.16236",
    sourceNote:
      "Verified: arXiv 2006.16236 published 2020-06-29. Class framing: drop softmax, regroup so state stays d×d.",
    problem:
      "Softmax forces a full token×token score matrix before mixing values. Memory and compute explode with n.",
    answer:
      "Replace softmax with a kernel feature map φ so Attention ≈ φ(Q)(φ(K)ᵀV). Associate into fixed-size state S = Σ φ(k)ᵀv — O(d²) per token.",
    buys: [
      "Constant-size recurrent state instead of growing KV cache",
      "Linear time in sequence length",
      "Unlocks the 'RNN-ify the Transformer' research line"
    ],
    costs: [
      "Loses softmax competition (non-negativity + sum-to-one sharpening)",
      "Every query reads the same compressed state — no fresh exact distribution over old keys",
      "Associative recall often lags dense softmax"
    ],
    when:
      "Extreme length / memory budgets where approximate mixing is OK. Often hybridized with a few softmax layers.",
    pickFor: "Million-token streams with fixed RAM",
    skipFor: "Tasks needing precise token-level lookback"
  },
  {
    id: "delta-rule",
    name: "Delta Rule / DeltaNet",
    short: "Delta rule",
    date: "2021-02-22",
    dateLabel: "Feb 2021",
    era: "memory",
    family: "linear",
    coveredInClass: true,
    paper: "Linear Transformers Are Secretly Fast Weight Programmers",
    authors: "Schlag, Irie, Schmidhuber",
    arxiv: "2102.11174",
    sourceNote:
      "Verified: arXiv 2102.11174 published 2021-02-22. Hardware-efficient parallel training: Yang et al. arXiv:2406.06484 published 2024-06-10.",
    problem:
      "Naive linear attention only adds into the state. New information piles on old (40+55→95) instead of overwriting what the model now wants (55).",
    answer:
      "Update associative memory with a delta / error-correcting rule: retrieve what the key currently maps to, write the difference toward the new value.",
    buys: [
      "Better overwriting and associative recall than pure additive linear attention",
      "Still a fixed-size state — memory bill stays flat in n",
      "Later parallel algorithms made it trainable at LLM scale (2024)"
    ],
    costs: [
      "Still not softmax-exact over full history",
      "Original sequential form was hard to scale until 2024 parallel work",
      "Extra state algebra vs vanilla linear attention"
    ],
    when:
      "Linear-time memory with stronger recall than additive kernels. Pair with gating (Gated DeltaNet) for modern stacks.",
    pickFor: "Long-context linear hybrids needing recall",
    skipFor: "Drop-in replacement for dense MHA quality"
  },
  {
    id: "rope",
    name: "Rotary Position Embedding (RoPE)",
    short: "RoPE",
    date: "2021-04-20",
    dateLabel: "Apr 2021",
    era: "length",
    family: "position",
    coveredInClass: true,
    paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding",
    authors: "Su et al.",
    arxiv: "2104.09864",
    sourceNote: "Verified: arXiv 2104.09864 published 2021-04-20.",
    problem:
      "Absolute PEs do not extrapolate well and do not make relative distance obvious. Attention needs 'how far apart?', not only 'what is my absolute ID?'.",
    answer:
      "Rotate Q and K in 2D subspaces by an angle proportional to position. The dot product depends on relative offset. No PE added to the residual stream.",
    buys: [
      "Relative distances fall out of the geometry",
      "No absolute position table to outgrow immediately",
      "Default PE for LLaMA-style models"
    ],
    costs: [
      "Long-range phases weaken; naive RoPE falters far past train length",
      "Needs scaling tricks (NTK, YaRN) or DroPE-style remedies for big jumps",
      "Interacts badly with some cache/window hacks if positions are mishandled"
    ],
    when:
      "Default for modern decoder LLMs in the trained context band. Plan an extension method if you promise ≫ train length.",
    pickFor: "Standard LLM pretraining up to designed L",
    skipFor: "Blind 32× context claims with plain RoPE"
  },
  {
    id: "alibi",
    name: "ALiBi (Attention with Linear Biases)",
    short: "ALiBi",
    date: "2021-08-27",
    dateLabel: "Aug 2021",
    era: "length",
    family: "position",
    coveredInClass: true,
    paper: "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation",
    authors: "Press, Smith, Lewis",
    arxiv: "2108.12409",
    sourceNote: "Verified: arXiv 2108.12409 published 2021-08-27.",
    problem:
      "Absolute PEs often failed when tested longer than trained. Position was still a length trap.",
    answer:
      "Skip explicit PE vectors. Add a head-specific linear penalty to attention scores proportional to distance — closer tokens win the bias.",
    buys: [
      "Strong zero-shot length extrapolation vs absolute PE of the era",
      "Tiny implementation change to the score matrix",
      "No rotary math / no position embedding parameters"
    ],
    costs: [
      "Hard-wired preference for locality — can undervalue long-range links",
      "Less widely adopted than RoPE in the LLaMA ecosystem",
      "Bias slopes are hyperparameters per head"
    ],
    when:
      "When extrapolation matters more than matching the RoPE ecosystem (e.g. some Bloom/MPT-style stacks).",
    pickFor: "Train-short / test-long LM research",
    skipFor: "RoPE-centric tooling & checkpoints"
  },
  {
    id: "flash-attention",
    name: "FlashAttention (Exact, IO-Aware)",
    short: "FlashAttention",
    date: "2022-05-27",
    dateLabel: "May 2022",
    era: "compute",
    family: "systems",
    coveredInClass: false,
    bonus: true,
    paper: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
    authors: "Dao et al.",
    arxiv: "2205.14135",
    sourceNote:
      "Verified: arXiv 2205.14135 published 2022-05-27. Bonus — same math as standard attention; pays less of the HBM bill.",
    problem:
      "Even exact attention was slow because naive kernels materialize the huge score matrix in HBM. The wall was IO, not asymptotics.",
    answer:
      "Tile attention in SRAM, never write the full n×n matrix to HBM, recompute in backward — exact softmax, fewer memory round-trips.",
    buys: [
      "Exact attention with large wall-clock and memory wins",
      "Made longer contexts practical without changing the model equation",
      "Infrastructure default (PyTorch SDPA, etc.)"
    ],
    costs: [
      "Does not change O(n²) asymptotics — only constants and memory traffic",
      "Kernel/hardware specific",
      "Easy to confuse with approximate attention — it is not"
    ],
    when:
      "Always, when available, under exact attention. Orthogonal to MQA/GQA/window/linear — stack them.",
    pickFor: "Any exact-attention training/serving stack",
    skipFor: "N/A as a model choice — it is an implementation"
  },
  {
    id: "gqa",
    name: "Grouped-Query Attention (GQA)",
    short: "GQA",
    date: "2023-05-22",
    dateLabel: "May 2023",
    era: "memory",
    family: "kv-sharing",
    coveredInClass: true,
    paper: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints",
    authors: "Ainslie et al.",
    arxiv: "2305.13245",
    sourceNote: "Verified: arXiv 2305.13245 published 2023-05-22.",
    problem:
      "MQA saved memory but sometimes hurt quality. Full MHA wasted bandwidth. Needed a middle ground and a way to convert existing checkpoints.",
    answer:
      "Share each KV head across a group of query heads (more than 1 KV head, fewer than Q heads). Uptrain from MHA with a small fraction of original compute.",
    buys: [
      "Near-MHA quality with near-MQA decode speed/memory",
      "Practical uptraining recipe from multi-head checkpoints",
      "LLM serving default (LLaMA-2/3-style)"
    ],
    costs: [
      "Still a compromise — not free vs full MHA",
      "Group factor is another hyperparameter",
      "Uptraining is cheap but not zero"
    ],
    when:
      "Almost always for modern LLM inference. Prefer over MQA unless extremely bandwidth starved.",
    pickFor: "Production decoder LLMs",
    skipFor: "Tiny models where MHA KV already fits"
  },
  {
    id: "ntk-aware",
    name: "NTK-Aware RoPE Scaling",
    short: "NTK-aware",
    date: "2023-06-30",
    dateLabel: "Jun 2023",
    era: "length",
    family: "rope-ext",
    coveredInClass: true,
    paper: "Community method (u/bloc97); documented in YaRN",
    authors: "bloc97 (community); recounted by Peng et al.",
    arxiv: "2309.00071",
    sourceNote:
      "No standalone arXiv at launch. Primary public date: 2023-06-30 — r/LocalLLaMA post by u/bloc97 (also mirrored same day in Hugging Face TGI issue #512; credited in transformers#24653). YaRN (arXiv:2309.00071, published 2023-08-31) is the canonical written account. Earlier estimate of mid-June was wrong; corrected to 2023-06-30.",
    problem:
      "Naive RoPE interpolation for longer context hurt high-frequency dimensions — local 'resolution' collapsed.",
    answer:
      "Scale the RoPE base so that as you stretch context, you preserve more high-frequency detail instead of uniformly interpolating all wavelengths.",
    buys: [
      "Better zero-shot / light-tune context extension than naive position interpolation alone",
      "Works as inference-time or light fine-tune tweak on RoPE models",
      "Direct ancestor of YaRN's 'NTK-by-parts' story"
    ],
    costs: [
      "Community origin — details varied across forks before YaRN",
      "Still not magic to arbitrary length",
      "Tied to RoPE; useless for ALiBi/absolute PE models"
    ],
    when:
      "Extending a RoPE model a few× beyond train length with minimal data. Prefer YaRN if you can fine-tune briefly.",
    pickFor: "Quick RoPE context stretch",
    skipFor: "Non-RoPE architectures"
  },
  {
    id: "yarn",
    name: "YaRN (Yet another RoPE extensioN)",
    short: "YaRN",
    date: "2023-08-31",
    dateLabel: "Aug 2023",
    era: "length",
    family: "rope-ext",
    coveredInClass: true,
    paper: "YaRN: Efficient Context Window Extension of Large Language Models",
    authors: "Peng et al.",
    arxiv: "2309.00071",
    sourceNote:
      "Verified: arXiv 2309.00071 <published> 2023-08-31 (id is 2309.* but first version landed Aug 31).",
    problem:
      "Position interpolation and NTK-aware each helped, but extending RoPE LLMs still needed too much compute or lost too much quality.",
    answer:
      "Combine NTK-by-parts interpolation, attention temperature scaling, and optional fine-tuning — extend context with ≪1% of original pretrain compute.",
    buys: [
      "Strong extended-context quality after tiny fine-tunes",
      "Dynamic scaling variants for inference-time stretch",
      "Standard recipe for RoPE long-context forks"
    ],
    costs: [
      "Still a RoPE patch, not a new attention law",
      "Fine-tune data/length choices matter; bad recipes hallucinate length",
      "Does not shrink the KV memory bill — only the PE length trap"
    ],
    when:
      "Taking a RoPE base (LLaMA/Mistral-class) to 32K–128K with limited continued training.",
    pickFor: "RoPE long-context fine-tunes",
    skipFor: "Cutting KV RAM (use GQA/MLA/window too)"
  },
  {
    id: "attention-sinks",
    name: "Attention Sinks (StreamingLLM)",
    short: "Attention sinks",
    date: "2023-09-29",
    dateLabel: "Sep 2023",
    era: "length",
    family: "sparse",
    coveredInClass: true,
    paper: "Efficient Streaming Language Models with Attention Sinks",
    authors: "Xiao et al.",
    arxiv: "2309.17453",
    sourceNote: "Verified: arXiv 2309.17453 published 2023-09-29.",
    problem:
      "Sliding-window KV should enable infinite streams, but perplexity collapses once the first tokens fall out — attention had been using them as a numerical 'sink'.",
    answer:
      "Keep a few initial tokens' KV forever (the sinks) plus a rolling window of recent tokens. Restore stable streaming without retraining.",
    buys: [
      "Infinite-stream decoding with roughly constant cache",
      "No fine-tune required for many RoPE/ALiBi models",
      "Explains a real failure mode of naive windowing"
    ],
    costs: [
      "Sinks are not semantic memory of the whole past — middle tokens still vanish",
      "Positional bookkeeping must be done carefully",
      "Does not restore true long-range access to evicted content"
    ],
    when:
      "Multi-turn / streaming apps that need unbounded sessions more than perfect recall of everything said hour one.",
    pickFor: "Streaming chat with fixed VRAM",
    skipFor: "Agents that must reread arbitrary early evidence"
  },
  {
    id: "mla",
    name: "Multi-Head Latent Attention (MLA)",
    short: "MLA",
    date: "2024-05-07",
    dateLabel: "May 2024",
    era: "memory",
    family: "compression",
    coveredInClass: true,
    paper: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model",
    authors: "DeepSeek-AI",
    arxiv: "2405.04434",
    sourceNote:
      "Verified: arXiv 2405.04434 published 2024-05-07. MLA = DeepSeek's compressed-KV attention (store low-rank latents, not fat K/V).",
    problem:
      "Even GQA's KV cache dominates serving cost at long context. Sharing heads helps, but you still store full K/V per token per KV head.",
    answer:
      "Compress keys/values into a low-rank latent cache; up-project when computing attention. Store the small latent instead of fat K/V tensors.",
    buys: [
      "Large KV-cache reduction vs MHA/GQA at similar quality (DeepSeek-V2 reports)",
      "Keeps softmax attention over the chosen representation",
      "Direct attack on the serving memory bill"
    ],
    costs: [
      "More complex projections; training recipe matters",
      "Compression can lose fine key distinctions",
      "Ecosystem / tooling less ubiquitous than GQA"
    ],
    when:
      "Long-context MoE serving where KV RAM is the limiter and you control the architecture.",
    pickFor: "Long-context economical serving",
    skipFor: "Tiny models / frameworks without MLA kernels"
  },
  {
    id: "gated-deltanet",
    name: "Gated DeltaNet",
    short: "Gated DeltaNet",
    date: "2024-12-09",
    dateLabel: "Dec 2024",
    era: "memory",
    family: "linear",
    coveredInClass: true,
    paper: "Gated Delta Networks: Improving Mamba2 with Delta Rule",
    authors: "Yang, Yang, Gu, et al.",
    arxiv: "2412.06464",
    sourceNote:
      "Verified: arXiv 2412.06464 published 2024-12-09. Builds on delta-rule (2021) and parallel DeltaNet (2024-06).",
    problem:
      "Delta-rule memory helps recall, but without gating the state struggles to forget / control write strength the way modern SSMs do.",
    answer:
      "Combine delta-rule updates with gating (Mamba2-inspired) so the fixed-size state can selectively write, forget, and retain.",
    buys: [
      "Stronger LM results among linear / hybrid recurrent lines",
      "Fixed state size — length-friendly memory bill",
      "Puts back control that removing softmax took away"
    ],
    costs: [
      "Still approximates full attention; hybrids often win",
      "Newer stack — tooling thinner than Transformers",
      "Parallel training algorithms are nontrivial"
    ],
    when:
      "Hybrids chasing million-token agents on fixed RAM with better recall than vanilla linear attention.",
    pickFor: "Long-agent hybrids on fixed state",
    skipFor: "When you need mature GQA+Flash ecosystem only"
  },
  {
    id: "deepseek-nsa",
    name: "DeepSeek Compressed Sparse Attention (NSA)",
    short: "DeepSeek NSA",
    date: "2025-02-16",
    dateLabel: "Feb 2025",
    era: "memory",
    family: "sparse",
    coveredInClass: true,
    paper: "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention",
    authors: "Yuan, Gao, Dai, et al. (DeepSeek & collaborators)",
    arxiv: "2502.11089",
    sourceNote:
      "Verified: arXiv 2502.11089 published 2025-02-16. NSA = hierarchical compressed tokens + fine-grained selection + windows — i.e. DeepSeek's research 'compressed sparse attention'. Distinct from MLA (2024-05, KV compression only) and from DSA (2025-09, production sparse on MLA).",
    problem:
      "You rarely need all past tokens — but dense softmax still scores all of them. Compression alone (MLA) does not skip irrelevant keys; naive sparsity is slow on GPUs.",
    answer:
      "Train sparse attention natively: coarse token compression for global context, fine-grained top-k-style selection for precision, plus local windows — with kernels aligned to hardware.",
    buys: [
      "Sub-quadratic work while retaining softmax on the kept set",
      "Compression + selection in one design (global + local)",
      "End-to-end trainable; reported speedups at 64k across train and decode"
    ],
    costs: [
      "If too sparse, you drop the one token that mattered",
      "Indexer / pattern errors are silent failure modes",
      "Kernel and training complexity above plain GQA"
    ],
    when:
      "Very long contexts where most history is irrelevant and you can train the selector end-to-end.",
    pickFor: "Long MoE / long-context pretrain with trained sparsity",
    skipFor: "Safety-critical full-context audit tasks"
  },
  {
    id: "deepseek-dsa",
    name: "DeepSeek Sparse Attention (DSA)",
    short: "DeepSeek DSA",
    date: "2025-09-29",
    dateLabel: "Sep 2025",
    era: "memory",
    family: "sparse",
    coveredInClass: false,
    bonus: true,
    paper: "DeepSeek-V3.2-Exp (DSA on MLA)",
    authors: "DeepSeek-AI",
    arxiv: null,
    sourceNote:
      "Primary public date: 2025-09-29 DeepSeek API docs announcement (api-docs.deepseek.com/news/news250929) + tech report in deepseek-ai/DeepSeek-V3.2-Exp. Production sparse attention: lightning indexer + top-k over MLA latents. Bonus — clarifies NSA (research) vs DSA (product).",
    problem:
      "MLA cut what you store; serving still scored too many latents per query at long context. Needed fine-grained sparsity on the compressed cache.",
    answer:
      "Add a lightweight indexer that picks top-k positions in the MLA latent cache, then run attention only on that subset — DSA in V3.2-Exp.",
    buys: [
      "Stacks selection on top of MLA compression (double discount)",
      "Production kernels (FlashMLA sparse) and reported API cost cuts",
      "Keeps MLA ecosystem rather than replacing the whole stack"
    ],
    costs: [
      "Depends on indexer quality — miss the needle and quality drops",
      "Tied to DeepSeek's MLA stack / kernels",
      "Very new; independent reproductions still catching up"
    ],
    when:
      "When you already run MLA and long-context decode cost is the product metric.",
    pickFor: "MLA serving at long context",
    skipFor: "Non-MLA architectures without a port"
  },
  {
    id: "drope",
    name: "DroPE (Drop Positional Embeddings)",
    short: "DroPE",
    date: "2025-12-13",
    dateLabel: "Dec 2025",
    era: "length",
    family: "position",
    coveredInClass: true,
    paper: "Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings",
    authors: "Gelberg, Eguchi, Akiba, Cetin (Sakana AI)",
    arxiv: "2512.12167",
    sourceNote:
      "Verified: arXiv 2512.12167 published 2025-12-13. Sakana AI report/blog at pub.sakana.ai/DroPE.",
    problem:
      "RoPE helps models converge in-distribution, but that same dependence blocks length generalization — PE scaling patches keep fighting the symptom.",
    answer:
      "Pretrain with RoPE as a scaffold, then drop positional embeddings and run a short recalibration at the original context length. Keep the training benefit; remove the extrapolation handcuff.",
    buys: [
      "Large zero-shot context extension without long-context fine-tuning (per paper)",
      "Preserves in-window quality after recalibration",
      "Reframes PE as a temporary inductive bias, not a permanent tax"
    ],
    costs: [
      "Requires a recalibration stage — not a free inference switch",
      "Very new; long-term ecosystem results still accumulating",
      "Does not by itself shrink KV cache size"
    ],
    when:
      "When you already have a RoPE-pretrained model and need length headroom without a huge long-context continued-pretrain.",
    pickFor: "Post-train length unlock on RoPE LMs",
    skipFor: "Cutting decode VRAM (combine with GQA/MLA)"
  }
];

window.ERAS = {
  exactness: {
    title: "Exactness first",
    blurb: "Pay full price for every pair. Get the right answer."
  },
  compute: {
    title: "Cut the multiply / IO bill",
    blurb: "Same idea, fewer edges or smarter IO."
  },
  memory: {
    title: "Cut the memory bill",
    blurb: "KV cache and state size become the enemy."
  },
  length: {
    title: "Buy length",
    blurb: "Positions and windows that survive past training L."
  }
};

/** Minimum assignment coverage checklist (ids). */
window.REQUIRED_IDS = [
  "scaled-dot-product",
  "learned-absolute",
  "sinusoidal",
  "rope",
  "alibi",
  "mqa",
  "gqa",
  "sliding-window",
  "attention-sinks",
  "ntk-aware",
  "yarn",
  "linear-attention",
  "delta-rule",
  "gated-deltanet",
  "mla",
  "sparse-topk",
  "deepseek-nsa",
  "drope"
];
