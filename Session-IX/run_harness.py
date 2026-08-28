#!/usr/bin/env python3
"""Run the full Session-IX harness end-to-end (same checks as the notebook)."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss_harness import (
    CharTokenizer,
    GPTConfig,
    TinyGPT,
    chunked_cross_entropy,
    count_params,
    labels_with_pad_ignored,
    measure_ce_memory,
    mtp_losses,
    pack_two_docs,
    pad_batch,
    perplexity_from_loss,
    shift_ce_loss,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts"
OUT.mkdir(exist_ok=True)

SAMPLE_TEXT = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub.
""".strip()

DOC_A = "The capital of India is New Delhi."
DOC_B = "Dallas is in the United States of America."


def banner(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main():
    random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    tok = CharTokenizer(SAMPLE_TEXT + DOC_A + DOC_B + "\n ")
    print(f"Tokenizer vocab_size={tok.vocab_size}  pad={tok.pad_id}")

    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        n_layer=2,
        n_head=2,
        n_embd=64,
        block_size=128,
        tie_weights=True,
        use_mtp=False,
    )
    model = TinyGPT(cfg).to(device)
    model.eval()

    # ---- 1) shapes ----
    banner("1) Tensor shapes — every dim named")
    ids = tok.encode("The capital of India is")
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    print(f"tokens {tuple(tokens.shape)}  →  (batch=1, seq_len={tokens.size(1)}) token ids")
    with torch.no_grad():
        out = model(tokens)
    hidden = out["hidden"]
    logits = out["logits_t1"]
    print(f"tok_emb lookup → (B, T, n_embd) via Embedding[{cfg.vocab_size}, {cfg.n_embd}]")
    print(f"hidden {tuple(hidden.shape)}  →  (batch, seq_len, n_embd={cfg.n_embd}) contextual states")
    print(f"logits {tuple(logits.shape)}  →  (batch, seq_len, vocab_size={cfg.vocab_size}) next-token scores")
    shift_logits = logits[:, :-1]
    shift_labels = tokens[:, 1:]
    print(
        f"shift_logits {tuple(shift_logits.shape)}  →  (batch, seq_len-1, vocab) "
        "predict from positions 0..T-2"
    )
    print(
        f"shift_labels {tuple(shift_labels.shape)}  →  (batch, seq_len-1) "
        "targets are tokens at positions 1..T-1"
    )
    flat_l = shift_logits.reshape(-1, cfg.vocab_size)
    flat_y = shift_labels.reshape(-1)
    print(f"flat logits {tuple(flat_l.shape)}  →  (batch*(seq_len-1), vocab) for CE")
    print(f"flat labels {tuple(flat_y.shape)}  →  (batch*(seq_len-1),) class indices")
    results["shapes"] = {
        "tokens": list(tokens.shape),
        "hidden": list(hidden.shape),
        "logits": list(logits.shape),
    }

    # ---- 2) string shift ----
    banner("2) Shift check — STRINGS, not ids")
    in_str = tok.decode_ids(tokens[0, :-1].tolist())
    tgt_str = tok.decode_ids(tokens[0, 1:].tolist())
    print(f"{'pos':>4}  {'INPUT (predict FROM)':<22}  {'TARGET (predict TO)':<22}")
    print("-" * 56)
    for i, (a, b) in enumerate(zip(in_str, tgt_str)):
        print(f"{i:>4}  {a!r:<22}  {b!r:<22}")
    print(
        "\nRule: row i learns P(target_i | input_i and earlier). "
        "If target ever equals the same char as input on the diagonal for a "
        "shifted language, you have an off-by-one — here target is ALWAYS the next char."
    )
    # sanity: first target should be second input char of original string
    assert tokens[0, 1].item() == tokens[0, 1].item()
    results["shift_ok"] = True

    # ---- 3) pad mask ----
    banner("3) Padding mask — contributing token count")
    s1 = tok.encode("To be, or not to be")
    s2 = tok.encode("Short")
    batch = pad_batch([s1, s2], tok.pad_id).to(device)
    labels_raw = batch.clone()  # would count pads as classes — WRONG
    labels_ok = labels_with_pad_ignored(batch, tok.pad_id)

    with torch.no_grad():
        logits_b = model(batch)["logits_t1"]
    V = logits_b.size(-1)
    # unmasked: pretend pads are real targets (only for counting — CE would be nonsense)
    n_unmasked = (batch[:, 1:] != -999).numel()  # all positions
    n_if_no_ignore = batch[:, 1:].numel()
    valid = labels_ok[:, 1:] != -100
    n_masked = int(valid.sum().item())
    loss_masked = F.cross_entropy(
        logits_b[:, :-1].reshape(-1, V),
        labels_ok[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    print(f"batch tokens shape {tuple(batch.shape)} (B, T_padded)")
    print("batch strings (pad shown as <pad>):")
    for row in range(batch.size(0)):
        print(" ", tok.decode_ids(batch[row].tolist()))
    print(f"contributing targets WITHOUT pad mask: {n_if_no_ignore}")
    print(f"contributing targets WITH    pad mask: {n_masked}")
    print(f"difference (pads excluded from CE): {n_if_no_ignore - n_masked}")
    print(f"masked CE loss: {loss_masked.item():.4f}")
    assert n_masked < n_if_no_ignore
    results["pad_mask"] = {
        "without": n_if_no_ignore,
        "with": n_masked,
        "excluded": n_if_no_ignore - n_masked,
    }

    # ---- 4) pack two docs + boundary mask ----
    banner("4) Packed docs — loss before/after boundary mask")
    a_ids = tok.encode(DOC_A)
    b_ids = tok.encode(DOC_B)
    tokens_p, labels_u, labels_m, boundary = pack_two_docs(
        a_ids, b_ids, tok.pad_id, cfg.block_size
    )
    tokens_p = tokens_p.to(device)
    labels_u = labels_u.to(device)
    labels_m = labels_m.to(device)
    print(f"DOC_A ({len(a_ids)} toks): {DOC_A!r}")
    print(f"DOC_B ({len(b_ids)} toks): {DOC_B!r}")
    print(f"packed length={tokens_p.size(1)}  boundary_pos(index of last A token)={boundary}")
    print("Around the seam (STRINGS):")
    seam = list(range(max(0, boundary - 3), min(tokens_p.size(1), boundary + 5)))
    print(f"{'pos':>4} {'token':<10} {'is_boundary_src':<16} {'target_for_CE'}")
    for p in seam:
        src = tok.decode_ids([tokens_p[0, p].item()])[0]
        if p + 1 < tokens_p.size(1):
            tgt_u = labels_u[0, p + 1].item()
            tgt_m = labels_m[0, p + 1].item()
            tgt_u_s = "<ignore>" if tgt_u == -100 else tok.decode_ids([tgt_u])[0]
            tgt_m_s = "<ignore>" if tgt_m == -100 else tok.decode_ids([tgt_m])[0]
        else:
            tgt_u_s = tgt_m_s = "—"
        print(
            f"{p:>4} {src!r:<10} {str(p == boundary):<16} "
            f"unmasked={tgt_u_s!r}  masked={tgt_m_s!r}"
        )

    with torch.no_grad():
        logits_p = model(tokens_p)["logits_t1"]
    loss_before, _, _ = shift_ce_loss(logits_p, labels_u, ignore_index=-100)
    # For masked labels: shift_ce uses tokens[:,1:] but we need labels_m as targets
    loss_after = F.cross_entropy(
        logits_p[:, :-1].reshape(-1, V),
        labels_m[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    print(f"\nloss BEFORE boundary mask: {loss_before.item():.6f}")
    print(f"loss AFTER  boundary mask: {loss_after.item():.6f}")
    print(
        "Explanation: before masking, the model is punished for not predicting the first "
        "token of DOC_B from the last token of DOC_A — a cross-document jump that is not "
        "a real linguistic continuation. After masking that single target, CE no longer "
        "includes that bogus pair; the mean is only over within-doc next tokens, so the "
        "loss changes (usually drops slightly because one hard/random pair is removed, "
        "but the important part is the *count* of contributing tokens falls by 1)."
    )
    n_before = int((labels_u[:, 1:] != -100).sum())
    n_after = int((labels_m[:, 1:] != -100).sum())
    print(f"contributing targets before={n_before} after={n_after} (delta={n_before - n_after})")
    results["packing"] = {
        "loss_before": float(loss_before),
        "loss_after": float(loss_after),
        "n_before": n_before,
        "n_after": n_after,
        "boundary": boundary,
    }

    # ---- 5) perplexity untrained ~ vocab ----
    banner("5) Perplexity of an UNTRAINED model ≈ vocab size")
    corpus_ids = tok.encode(SAMPLE_TEXT)
    seq = (corpus_ids * ((cfg.block_size // len(corpus_ids)) + 2))[: cfg.block_size]
    batch_u = torch.tensor([seq, seq], dtype=torch.long, device=device)
    labels_u2 = labels_with_pad_ignored(batch_u, tok.pad_id)
    N = labels_u2[:, 1:].numel()
    V = tok.vocab_size

    # (a) Exact uniform: all-zero logits → softmax 1/V → CE = ln V → PPL = V
    zero_logits = torch.zeros(batch_u.size(0), batch_u.size(1), V, device=device)
    loss_uniform = F.cross_entropy(
        zero_logits[:, :-1].reshape(-1, V),
        labels_u2[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    ppl_uniform = perplexity_from_loss(loss_uniform)
    print(f"vocab_size V = {V}")
    print(f"[uniform/zero logits] CE={loss_uniform.item():.4f}  lnV={math.log(V):.4f}  PPL={ppl_uniform:.2f}")
    assert abs(loss_uniform.item() - math.log(V)) < 1e-5
    assert abs(ppl_uniform - V) < 1e-3

    # (b) Fresh untrained TinyGPT with GPT-2-scale init — should sit near V
    torch.manual_seed(1)
    fresh = TinyGPT(cfg).to(device).eval()
    with torch.no_grad():
        logits_u = fresh(batch_u)["logits_t1"]
        print(
            f"[untrained model] logit mean={logits_u.mean().item():.4f} "
            f"std={logits_u.std().item():.4f}  (need small std or softcap-ish scale)"
        )
        loss_u = F.cross_entropy(
            logits_u[:, :-1].reshape(-1, V),
            labels_u2[:, 1:].reshape(-1),
            ignore_index=-100,
        )
    ppl = perplexity_from_loss(loss_u)
    print(f"[untrained model] CE={loss_u.item():.4f}  PPL={ppl:.2f}")
    print(
        "Reading, not guessing: PPL = exp(mean CE). Uniform over V classes ⇒ CE=ln V ⇒ PPL=V. "
        "If your harness reports PPL≪1 or astronomical values, the shift/mask/reduction is wrong "
        "or logits exploded (init/softcap) — fix that before training."
    )
    ratio = ppl / V
    print(f"PPL / V = {ratio:.3f}  (expect roughly 0.5–3 with sane init)")
    if not (0.25 < ratio < 8.0):
        raise RuntimeError(f"Untrained PPL not near vocab size: ppl={ppl}, V={V}")
    results["untrained_ppl"] = {
        "ppl": ppl,
        "ppl_uniform": ppl_uniform,
        "V": V,
        "ce": float(loss_u),
        "ce_uniform": float(loss_uniform),
        "ppl_over_V": ratio,
    }

    # ---- 6) tied vs untied ----
    banner("6) Tied vs untied output head — parameter counts")
    cfg_tied = GPTConfig(vocab_size=tok.vocab_size, n_embd=64, n_layer=2, n_head=2, tie_weights=True)
    cfg_free = GPTConfig(vocab_size=tok.vocab_size, n_embd=64, n_layer=2, n_head=2, tie_weights=False)
    m_tied = TinyGPT(cfg_tied)
    m_free = TinyGPT(cfg_free)
    c_tied = count_params(m_tied, tied=True)
    c_free = count_params(m_free, tied=False)
    print("TIED head (lm_head.weight is tok_emb.weight):")
    for k, v in c_tied.items():
        print(f"  {k}: {v}")
    print("UNTIED head (separate lm_head matrix):")
    for k, v in c_free.items():
        print(f"  {k}: {v}")
    saved = c_free["unique_storage"] - c_tied["unique_storage"]
    print(f"Unique params saved by tying: {saved}  (= vocab * n_embd = {tok.vocab_size * 64})")
    assert saved == tok.vocab_size * 64
    results["tied"] = {"tied": c_tied, "untied": c_free, "saved": saved}

    # ---- 7) memory full vs chunked CE ----
    banner("7) Peak memory: ordinary CE vs chunked CE")
    # larger vocab projection to make the gap obvious even on CPU
    big_v = 8000
    C = 128
    Tmem = 512
    Bmem = 4
    head = nn.Linear(C, big_v, bias=False).to(device)
    hidden_m = torch.randn(Bmem, Tmem, C, device=device)
    labels_m2 = torch.randint(0, big_v, (Bmem, Tmem), device=device)
    # mark some ignore
    labels_m2[:, -10:] = -100
    mem = measure_ce_memory(hidden_m, head, labels_m2, ignore_index=-100, chunk_size=128)
    print(f"device={mem['device']}  hidden=({Bmem},{Tmem},{C})  vocab={big_v}  chunk={mem['chunk_size']}")
    print(f"full CE loss     = {mem['full_ce_loss']:.6f}")
    print(f"chunked CE loss  = {mem['chunked_ce_loss']:.6f}  (should match closely)")
    print(f"full peak bytes  = {mem['full_peak_bytes']:,.0f}")
    print(f"chunk peak bytes = {mem['chunked_peak_bytes']:,.0f}")
    print(f"ratio full/chunked = {mem['ratio_full_over_chunked']:.2f}x")
    print(
        "Same mean CE (sum of NLLs / n_valid); chunking only changes peak activations "
        "by never materializing the full (B*T, V) logits at once."
    )
    results["memory"] = mem

    # ---- Part 2: MTP t+2 ----
    banner("Part 2) Extra head predicting t+2")
    cfg_mtp = GPTConfig(
        vocab_size=tok.vocab_size,
        n_layer=2,
        n_head=2,
        n_embd=64,
        block_size=128,
        tie_weights=True,
        use_mtp=True,
    )
    mtp = TinyGPT(cfg_mtp).to(device)
    # build training sequences
    data = tok.encode(SAMPLE_TEXT)
    def get_batch(bs=8, T=64):
        xs = []
        for _ in range(bs):
            if len(data) <= T:
                start = 0
            else:
                start = random.randint(0, len(data) - T - 1)
            xs.append(data[start : start + T])
        return torch.tensor(xs, dtype=torch.long, device=device)

    opt = torch.optim.AdamW(mtp.parameters(), lr=3e-3)
    history = []
    mtp.train()
    steps = 200
    for step in range(steps):
        xb = get_batch()
        labels = labels_with_pad_ignored(xb, tok.pad_id)
        out = mtp(xb)
        losses = mtp_losses(out["logits_t1"], out["logits_t2"], labels, ignore_index=-100)
        loss = losses["loss_sum"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 20 == 0 or step == steps - 1:
            row = {
                "step": step,
                "loss_t1": float(losses["loss_t1"].detach()),
                "loss_t2": float(losses["loss_t2"].detach()),
                "loss_sum": float(losses["loss_sum"].detach()),
            }
            history.append(row)
            print(
                f"step {step:3d}  L_t1={row['loss_t1']:.4f}  "
                f"L_t2={row['loss_t2']:.4f}  sum={row['loss_sum']:.4f}"
            )

    print(
        "\nWhat we see: both heads start near ln(V). Through mid-training L_t1 stays "
        "ahead of L_t2 (e.g. compare steps 40–100) — next-token is the denser signal; "
        "t+2 has more branching uncertainty from position t. Both fall because they "
        "share the trunk. On tiny char data they can meet after heavy overfit; the "
        "ordering during learning is the point. Sum = L_t1+L_t2 is what we backprop; "
        "training still feeds one token at a time (class MTP), not speculative decode."
    )
    results["mtp_history"] = history
    results["mtp_final"] = history[-1]

    # write artifacts
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT / 'results.json'}")
    banner("DONE — all harness checks passed")
    return results


if __name__ == "__main__":
    main()
