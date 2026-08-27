"""Why does olmo3-l32-131k reproduce Neuronpedia at only r~0.90 / 20% error?

One model load, then a grid over the plausible causes:
  encode mode : stored-threshold vs per-token top-k (k=160)
  prefix      : window as-is vs prepending eos (Olmo has no BOS; <|endoftext|>
                is its document separator, and the windows are mid-document)
  ratio shape : per-case median(mine/exp) -- a constant != 1 across cases would
                indicate a scale factor (activation normalization, folded
                decoder norms); noise centered on 1 would not.
"""
import sys

import numpy as np
import requests
import torch

sys.path.insert(0, "/home/a5k/cwtice.a5k/sae-exploration/scripts")
from sae_local import REGISTRY, Runner  # noqa: E402

NAME = "olmo3-l32-131k"
cfg = REGISTRY[NAME]
runner = Runner(NAME)
tok = runner.tokenizer
print("threshold scalar:", runner.sae.threshold, " k:", runner.sae.k)
print("eos:", tok.eos_token, tok.eos_token_id, "| pad:", tok.pad_token)

# ground truth
cases = []
latent = 0
while len(cases) < 10 and latent < 40000:
    latent += 3137
    r = requests.get(f"https://www.neuronpedia.org/api/feature/olmo-3-1125-32b/"
                     f"32-res-batchtopk-131k/{latent}", timeout=60)
    if r.status_code != 200:
        continue
    for a in (r.json().get("activations") or []):
        if a.get("maxValue", 0) <= 0:
            continue
        ids = []
        ok = True
        for t in a["tokens"]:
            enc = tok.encode(t, add_special_tokens=False)
            if len(enc) != 1:
                ok = False
                break
            ids.append(enc[0])
        if not ok:
            continue
        ref = {i: float(v) for i, v in enumerate(a["values"]) if v > 0}
        if len(ref) >= 3:
            cases.append((latent, np.array(ids, dtype=np.int64), ref))
        break
print(f"{len(cases)} cases\n")

eos = tok.eos_token_id


def run(ids, mode, prefix):
    ids_run = np.concatenate([[eos], ids]) if prefix else ids
    shift = 1 if prefix else 0
    t = torch.tensor(ids_run)[None]
    h = runner.residual(t)
    saved = runner.sae.threshold
    if mode == "topk":
        runner.sae.threshold = None
    acts = runner.sae.encode(h)
    runner.sae.threshold = saved
    return acts[0], shift


print(f"{'mode':>6} {'prefix':>7} {'mean r':>8} {'med |rel err|':>14} {'med ratio':>10} "
      f"{'per-case ratios'}")
for mode in ["thresh", "topk"]:
    for prefix in [False, True]:
        rs, errs, ratios = [], [], []
        for latent, ids, ref in cases:
            acts, shift = run(ids, mode, prefix)
            got = acts[:, latent].float().cpu().numpy()
            keep = [q for q in ref if q + shift < len(got)]
            exp = np.array([ref[q] for q in keep])
            mine = np.array([got[q + shift] for q in keep])
            if exp.std() > 0 and mine.std() > 0:
                rs.append(np.corrcoef(mine, exp)[0, 1])
            errs.append(np.median(np.abs(mine - exp) / exp))
            ratios.append(np.median(mine / np.maximum(exp, 1e-9)))
        print(f"{mode:>6} {str(prefix):>7} {np.mean(rs):>8.4f} {np.median(errs):>13.1%} "
              f"{np.median(ratios):>10.3f} "
              f"{'[' + ', '.join(f'{x:.2f}' for x in ratios[:6]) + ']'}")
