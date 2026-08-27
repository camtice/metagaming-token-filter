"""Diagnose gemma-3 local-vs-reference activation mismatch.

Hypothesis under test: JumpReLU has a hard threshold, so tokens whose
pre-activation sits near the threshold flip on/off under bf16 rounding, producing
huge *relative* error on small activations while leaving large activations (the
ones that matter for labeling) essentially exact. If true, error should be
strongly concentrated at low activation magnitude and the top activations should
agree closely.
"""
import glob
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, "/home/a5k/cwtice.a5k/sae-exploration/scripts")
from sae_local import REGISTRY, Runner, sae_dir  # noqa: E402

NAME = sys.argv[1] if len(sys.argv) > 1 else "gemma3-l46-16k"
cfg = REGISTRY[NAME]
runner = Runner(NAME)

with safe_open(f"{sae_dir(NAME)}/examples.safetensors", framework="np") as f:
    freqs = f.get_slice("feature_frequencies")[:]
    seq_t, pos_t, act_t = (f.get_slice("seq_ids"), f.get_slice("positions"),
                           f.get_slice("activations"))
    toks_t = f.get_slice("tokens")
    order = np.argsort(-freqs)
    latents = [int(x) for x in order[len(order)//20::len(order)//60][:6]]

    pairs = []
    for latent in latents:
        seq_ids, positions, acts = seq_t[latent], pos_t[latent], act_t[latent]
        valid = seq_ids >= 0
        if not valid.any():
            continue
        sid = int(seq_ids[np.flatnonzero(valid)[np.argmax(acts[valid])]])
        same = [j for j in np.flatnonzero(valid) if int(seq_ids[j]) == sid]
        ids = np.asarray(toks_t[sid], dtype=np.int64)
        t = torch.tensor(ids, device=runner.device)[None]
        got = runner.sae.encode(runner.residual(t))[0, :, latent].float().cpu().numpy()
        for j in same:
            pairs.append((latent, float(acts[j]), float(got[int(positions[j])])))

pairs = np.array([(e, g) for _, e, g in pairs])
exp, mine = pairs[:, 0], pairs[:, 1]
thr = runner.sae.threshold.float().cpu().numpy()

print(f"\n{NAME}: {len(pairs)} reference points")
print(f"overall pearson r = {np.corrcoef(mine, exp)[0,1]:.5f}")
print(f"threshold range over dictionary: {thr.min():.3f} .. {thr.max():.3f} "
      f"(median {np.median(thr):.3f})")

print(f"\n{'exp act quantile':>18} {'n':>4} {'median |rel err|':>17} {'median exp':>11} "
      f"{'zeroed locally':>15}")
qs = np.quantile(exp, [0, .25, .5, .75, .9, 1.0])
for lo, hi in zip(qs[:-1], qs[1:]):
    m = (exp >= lo) & (exp <= hi)
    if m.sum() == 0:
        continue
    rel = np.abs(mine[m] - exp[m]) / np.maximum(exp[m], 1e-9)
    print(f"{lo:8.1f}-{hi:8.1f} {m.sum():>4} {np.median(rel):>16.1%} "
          f"{np.median(exp[m]):>11.1f} {(mine[m] == 0).sum():>15}")

top = exp >= np.quantile(exp, 0.75)
rel_top = np.abs(mine[top] - exp[top]) / exp[top]
print(f"\ntop-quartile activations: median rel err {np.median(rel_top):.2%}, "
      f"r = {np.corrcoef(mine[top], exp[top])[0,1]:.5f}")
print(f"ratio mine/exp on top quartile: median {np.median(mine[top]/exp[top]):.4f} "
      f"(1.0 = perfect; a constant != 1 would mean a scale bug)")
