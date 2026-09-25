"""Check RevStack gradients and reconstruction, with float vs fixed-point states.

For each reversible variant and two weight regimes ("init" and "grown", where all
linear weights are scaled 4x to mimic a model partway through training):
  - fp32: reversible backward vs plain autograd (float state) -> should agree to ~1e-6
  - fp16 autocast: error of each gradient against the fp32 ground truth for
      plain autograd | reversible with float state | reversible with fixed-point state
  - relative error of the embedding input rebuilt after reversing all layers
"""

import json
import os

import torch

from model import GPT, GPTConfig

HERE = os.path.dirname(os.path.abspath(__file__))


def grads(model, idx, tgt, amp):
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        loss = model(idx, tgt)
    loss.backward()
    return loss.item(), {n: p.grad.detach().clone() for n, p in model.named_parameters()}


def rel_err(g, truth):
    num = sum(((g[n] - truth[n]) ** 2).sum() for n in truth)
    den = sum((truth[n] ** 2).sum() for n in truth)
    return (num / den).sqrt().item()


def make(mode, T, sd, **kw):
    m = GPT(GPTConfig(mode=mode, block_size=T, **kw)).cuda()
    if sd is not None:
        m.load_state_dict(sd)
    return m


def main():
    torch.manual_seed(0)
    B, T = 4, 256
    idx = torch.randint(0, 8192, (B, T), device="cuda")
    tgt = torch.randint(0, 8192, (B, T), device="cuda")
    out = {}
    for regime in ("init", "grown"):
        for mode in ("midpoint", "midpoint_a", "leapfrog", "euler"):
            ref = make(mode, T, None, rev_backward=False)
            if regime == "grown":
                with torch.no_grad():
                    for n, p in ref.named_parameters():
                        if p.dim() == 2 and "wte" not in n and "wpe" not in n:
                            p.mul_(4.0)
            sd = ref.state_dict()
            rev_f = make(mode, T, sd, exact=False)
            rev_x = make(mode, T, sd, exact=True)
            rev_f.track_recon = rev_x.track_recon = True

            _, truth = grads(ref, idx, tgt, amp=False)
            _, rev32 = grads(rev_f, idx, tgt, amp=False)
            _, naive16 = grads(ref, idx, tgt, amp=True)
            _, revf16 = grads(rev_f, idx, tgt, amp=True)
            recon_f = rev_f.last_recon_err
            _, revx16 = grads(rev_x, idx, tgt, amp=True)
            recon_x = rev_x.last_recon_err

            r = {
                "fp32_rev_vs_autograd": rel_err(rev32, truth),
                "fp16_autograd_vs_fp32": rel_err(naive16, truth),
                "fp16_rev_float_vs_fp32": rel_err(revf16, truth),
                "fp16_rev_fixed_vs_fp32": rel_err(revx16, truth),
                "recon_rel_err_float": recon_f,
                "recon_rel_err_fixed": recon_x,
                "fixed_point_used": rev_x.exact,
            }
            out[f"{regime}/{mode}"] = r
            print(f"{regime:5s} {mode:11s} fp32 rev-vs-autograd {r['fp32_rev_vs_autograd']:.1e} | fp16 grad err vs fp32: "
                  f"autograd {r['fp16_autograd_vs_fp32']:.2e}, rev-float {r['fp16_rev_float_vs_fp32']:.2e}, "
                  f"rev-fixed {r['fp16_rev_fixed_vs_fp32']:.2e} | recon err float {recon_f:.1e}, fixed {recon_x:.1e}")
    os.makedirs(os.path.join(HERE, "artifacts"), exist_ok=True)
    with open(os.path.join(HERE, "artifacts", "grad_check.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
