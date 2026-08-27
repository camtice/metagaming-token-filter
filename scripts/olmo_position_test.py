"""Missing-left-context hypothesis: if Neuronpedia computed Olmo activations on
2048-token sequences and shows 127-token excerpts, re-running the excerpt alone
should err most at early positions (little context recovered) and least at late
positions. Buckets relative error by token position within the window.

Contrast case: gemma-2, whose windows reproduced at 0.45% error, i.e. its
dashboards were computed on the shown window exactly.
"""
import sys

import numpy as np
import requests
import torch

sys.path.insert(0, "/home/a5k/cwtice.a5k/sae-exploration/scripts")
from sae_local import Runner  # noqa: E402

runner = Runner("olmo3-l32-131k")
tok = runner.tokenizer

pairs = []  # (position, exp, mine)
latent = 0
n_cases = 0
while n_cases < 12 and latent < 60000:
    latent += 2531
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
        if len(ref) < 3:
            continue
        acts = runner.sae.encode(runner.residual(torch.tensor(ids)[None]))[0]
        got = acts[:, latent].float().cpu().numpy()
        for q, v in ref.items():
            if q < len(got):
                pairs.append((q, v, float(got[q])))
        n_cases += 1
        break

pairs = np.array(pairs)
print(f"{n_cases} cases, {len(pairs)} reference points")
print(f"\n{'position bucket':>16} {'n':>5} {'median |rel err|':>17} {'median ratio':>13}")
for lo, hi in [(0, 16), (16, 32), (32, 64), (64, 96), (96, 128)]:
    m = (pairs[:, 0] >= lo) & (pairs[:, 0] < hi)
    if m.sum() < 3:
        print(f"{f'{lo}-{hi}':>16} {m.sum():>5}   (too few)")
        continue
    exp, mine = pairs[m, 1], pairs[m, 2]
    rel = np.abs(mine - exp) / exp
    print(f"{f'{lo}-{hi}':>16} {m.sum():>5} {np.median(rel):>16.1%} "
          f"{np.median(mine/exp):>13.3f}")
