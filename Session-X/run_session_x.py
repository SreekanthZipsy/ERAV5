#!/usr/bin/env python3
"""Run every Session-X check; write artifacts/results.json + figures."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from truth_loop import (
    Config,
    TinyLM,
    causal_ce,
    describe_0_1,
    estimate_mfu,
    trace_training_step,
    train_accum_comparison,
    train_with_grad_norm_log,
)

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
FIG = ROOT / "figures"
ART.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)


def banner(t: str) -> None:
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


def main() -> dict:
    results: dict = {}
    device = torch.device("cpu")
    torch.manual_seed(0)

    # ------------------------------------------------------------------
    # 1) Shapes in one step
    # ------------------------------------------------------------------
    banner("1) Every tensor shape in one training step")
    cfg = Config(vocab_size=50, n_embd=64, n_layer=2, n_head=2, block_size=32)
    model = TinyLM(cfg).to(device)
    B, T = 2, 24
    tokens = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    shape_rows = trace_training_step(model, tokens)
    results["shapes"] = {
        "tokens": list(tokens.shape),
        "n_tensors_reported": len(shape_rows),
        "rows": shape_rows,
        "n_valid": B * (T - 1),
        "n_params": model.n_params(),
    }

    # ------------------------------------------------------------------
    # 2) Hand gradient check
    # ------------------------------------------------------------------
    banner("2) Hand-verify one gradient (central difference vs backward)")
    from truth_loop import verify_grad_hand

    vg = verify_grad_hand(eps=1e-3)
    print("Toy L=(w*x-t)^2:")
    print(f"  analytic={vg['toy']['analytic']:.12f}")
    print(f"  numerical={vg['toy']['numerical']:.12f}")
    print(f"  |Δ|={vg['toy']['abs_err']:.2e}")
    print("TinyLM CE (float64, mlp weight):")
    print(f"  w={vg['tinylm_ce']['original_weight']:.10f}, eps={vg['tinylm_ce']['eps']}")
    print(f"  L(w+eps)={vg['tinylm_ce']['loss_plus']:.12f}")
    print(f"  L(w-eps)={vg['tinylm_ce']['loss_minus']:.12f}")
    print(f"  analytic={vg['tinylm_ce']['analytic']:.10f}")
    print(f"  numerical={vg['tinylm_ce']['numerical']:.10f}")
    print(f"  |Δ|={vg['tinylm_ce']['abs_err']:.2e}  rel={vg['tinylm_ce']['rel_err']:.2e}")
    assert vg["toy"]["abs_err"] < 1e-8
    assert vg["tinylm_ce"]["abs_err"] < 5e-5 or vg["tinylm_ce"]["rel_err"] < 1e-3
    results["grad_check"] = vg
    print("Agree to several decimals — autograd matches the definition of the derivative.")

    # ------------------------------------------------------------------
    # 3) Break gradient accumulation on purpose
    # ------------------------------------------------------------------
    banner("3) Broken accum (mean of means) vs token-weighted — plot the gap")
    hist = train_accum_comparison(steps=80, seed=0)
    results["accum"] = {
        "final_token_eval_wrong_model": hist["eval_wrong_model"][-1],
        "final_token_eval_right_model": hist["eval_right_model"][-1],
        "final_short_eval_wrong_model": hist["short_wrong_model"][-1],
        "final_short_eval_right_model": hist["short_right_model"][-1],
        "mean_objective_gap": sum(hist["objective_gap_initial_model"]) / len(hist["objective_gap_initial_model"]),
        "lengths": [16, 16, 48, 48],
        "valid_tokens": hist["valid_tokens"],
        "note": "Wrong = equal vote per microbatch mean; Right = sum(nll)/sum(n).",
    }
    print(f"microbatch lengths: {results['accum']['lengths']}")
    print(f"valid next-token counts: {hist['valid_tokens']} (total={sum(hist['valid_tokens'])})")
    print("Wrong weights each microbatch 25%; correct token weights are "
          + ", ".join(f"{100*n/sum(hist['valid_tokens']):.1f}%" for n in hist["valid_tokens"]))
    print(f"final common token-weighted eval, wrong-trained: {hist['eval_wrong_model'][-1]:.4f}")
    print(f"final common token-weighted eval, right-trained: {hist['eval_right_model'][-1]:.4f}")
    print(f"mean pure objective gap: {results['accum']['mean_objective_gap']:.6f}")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(hist["step"], hist["eval_wrong_model"], label="wrong-trained model")
        ax[0].plot(hist["step"], hist["eval_right_model"], label="token-weighted model")
        ax[0].set_xlabel("step")
        ax[0].set_ylabel("common token-weighted eval loss")
        ax[0].set_title("Same metric, different updates")
        ax[0].legend()
        ax[1].plot(hist["step"], hist["objective_gap_initial_model"], color="#c45c26")
        ax[1].axhline(0, color="#888", lw=0.8)
        ax[1].set_xlabel("step")
        ax[1].set_ylabel("mean-of-means − token-weighted")
        ax[1].set_title("Objective gap on identical data/model")
        fig.tight_layout()
        fig.savefig(FIG / "accum_gap.png", dpi=140)
        plt.close(fig)
        print(f"wrote {FIG / 'accum_gap.png'}")
    except Exception as e:
        print("plot skipped:", e)

    # ------------------------------------------------------------------
    # 4) Grad norm log — find jump before loss
    # ------------------------------------------------------------------
    banner("4) Grad norm every step — find where grad moved before loss")
    gn_hist = train_with_grad_norm_log(steps=120, spike_at=45)
    f = gn_hist["finding"]
    results["grad_norm"] = {
        "finding": f,
        "controlled_backward_scale_step": 45,
        "backward_scale": gn_hist["backward_scale"][45],
    }
    print("controlled failure: at step 45 only the backward objective is scaled 80x")
    print(f"grad_norm step 44 -> 45: {f['grad_norm_before']:.4f} -> {f['grad_norm_at_warning']:.4f}")
    print(f"probe loss (pre-update) step 44 -> 45 -> 46: "
          f"{f['probe_loss_before']:.4f} -> {f['probe_loss_at_warning_pre_update']:.4f} "
          f"-> {f['probe_loss_next_step']:.4f}")
    print(f["note"])

    try:
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(gn_hist["step"], gn_hist["probe_loss_pre_update"], color="#0d5c63", label="probe loss")
        ax1.set_xlabel("step")
        ax1.set_ylabel("probe loss", color="#0d5c63")
        ax2 = ax1.twinx()
        ax2.plot(
            gn_hist["step"],
            gn_hist["grad_norm"],
            color="#c45c26",
            alpha=0.85,
            label="grad_norm",
        )
        ax2.set_ylabel("grad norm", color="#c45c26")
        ax1.axvline(45, color="#888", ls="--", lw=1)
        ax1.set_title("Pre-update logging: grad warning at s, loss damage at s+1")
        fig.tight_layout()
        fig.savefig(FIG / "grad_norm_vs_loss.png", dpi=140)
        plt.close(fig)
        print(f"wrote {FIG / 'grad_norm_vs_loss.png'}")
    except Exception as e:
        print("plot skipped:", e)

    # ------------------------------------------------------------------
    # 5) MFU
    # ------------------------------------------------------------------
    banner("5) Honest MFU")
    cfg_m = Config(vocab_size=64, n_embd=128, n_layer=4, n_head=4, block_size=128)
    m = TinyLM(cfg_m).to(device)
    B_m, T_m = 8, 64
    tokens_m = torch.randint(0, cfg_m.vocab_size, (B_m, T_m), device=device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    # warmup
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss, _ = causal_ce(m(tokens_m)["logits"], tokens_m)
        loss.backward()
        opt.step()
    n_steps = 20
    t0 = time.perf_counter()
    tokens_done = 0
    for _ in range(n_steps):
        opt.zero_grad(set_to_none=True)
        loss, n = causal_ce(m(tokens_m)["logits"], tokens_m)
        loss.backward()
        opt.step()
        tokens_done += B_m * (T_m - 1)  # contributing approx
    wall = time.perf_counter() - t0
    n_params = m.n_params()
    # This VM exposes two physical Skylake/Cascade Lake cores. Nominal FP32 peak:
    # 2 cores × 2 AVX-512 FMA units × 16 lanes × 2 FLOP/FMA × 2.5 GHz = 0.320 TFLOP/s.
    # It is an upper bound: AVX frequency can be lower and a VM does not guarantee a whole core.
    peak_cpu_fp32 = 0.320
    mfu_window = estimate_mfu(
        n_params, tokens_done, wall, peak_cpu_fp32,
        n_layer=cfg_m.n_layer, n_embd=cfg_m.n_embd, seq_len=T_m
    )
    print(f"params (unique) = {n_params:,}")
    print(f"tokens processed = {tokens_done:,} in {wall:.3f}s ({n_steps} steps)")
    print("FLOPs = tokens × (6N + 12·layers·width·sequence_length)")
    print(f"model FLOPs ≈ {mfu_window['approx_flops']/1e9:.2f} GFLOP")
    print(f"achieved ≈ {mfu_window['achieved_tflops']:.4f} TFLOP/s")
    print(f"MFU proxy vs nominal CPU FP32 peak ({peak_cpu_fp32:.3f} TFLOP/s) = {mfu_window['mfu_pct']:.2f}%")
    print(
        "The gap to 40% is primarily tiny matrices, eager/Python launch overhead, unfused "
        "attention and cross entropy, Adam memory traffic, and VM scheduling. This CPU number "
        "is a transparent proxy, not an H100 MFU claim."
    )
    results["mfu"] = {
        "n_params": n_params,
        "tokens": tokens_done,
        "wall_s": wall,
        "approx_flops": mfu_window["approx_flops"],
        "achieved_tflops": mfu_window["achieved_tflops"],
        "mfu_pct_vs_nominal_cpu_fp32": mfu_window["mfu_pct"],
        "peak_cpu_fp32_tflops": peak_cpu_fp32,
        "peak_derivation": "2 cores * 2 AVX-512 FMA units * 16 fp32 lanes * 2 flop/FMA * 2.5 GHz",
        "why_not_40pct": (
            "Tiny matrices do not sustain peak SIMD throughput; eager Python and many small "
            "operators add overhead; Adam is memory-heavy; attention and CE are not fused; "
            "the virtualized CPU may be scheduled or AVX-downclocked."
        ),
        "caveat": "Nominal CPU peak is an architectural upper bound; this is an MFU proxy.",
    }

    # ------------------------------------------------------------------
    # 6) 0.1 in fp32 / bf16 / fp8 E4M3
    # ------------------------------------------------------------------
    banner("6) The number 0.1 — bits by hand")
    bits = describe_0_1()
    for name, d in bits.items():
        print(f"\n{name}: {d['layout']}")
        print(f"  bits: {d['bits']}   hex: {d['hex']}")
        print(
            f"  sign={d['sign']}  exp={d.get('exponent_bits')} "
            f"(unbiased {d.get('exponent_unbiased')})  mant={d.get('mantissa_bits')}"
        )
        print(f"  decodes ≈ {d['approx_value']}")
        if "rel_error_pct" in d:
            print(f"  relative error vs 0.1: {d['rel_error_pct']:.2f}%")
    print(
        "\nWhich would I train in? BF16 for the main compute (same exponent range as FP32, "
        "fits 2× vs FP32 bandwidth, safe for Adam's mixed small/large grads). Keep an FP32 "
        "master copy + optimizer state. FP8 E4M3 is great for matmuls with scaling on "
        "Blackwell/Hopper Transformer Engine, but 0.1 already has percent-level error and tiny "
        "grads need care — not my default for a first from-scratch run without TE."
    )
    results["float_0_1"] = bits
    results["train_dtype_choice"] = "bf16_compute_fp32_master"

    (ART / "results.json").write_text(json.dumps(results, indent=2))
    # also save series for notebook
    (ART / "accum_hist.json").write_text(json.dumps(hist))
    (ART / "grad_norm_hist.json").write_text(
        json.dumps(
            {
                "step": gn_hist["step"],
                "probe_loss_pre_update": gn_hist["probe_loss_pre_update"],
                "raw_train_loss": gn_hist["raw_train_loss"],
                "backward_scale": gn_hist["backward_scale"],
                "grad_norm": gn_hist["grad_norm"],
                "finding": gn_hist["finding"],
            }
        )
    )
    print(f"\nWrote {ART / 'results.json'}")
    banner("DONE")
    return results


if __name__ == "__main__":
    main()
