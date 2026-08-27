"""Do the explanation-classified latents fire on real metagaming traces at all?

The activation probe and the explanation-based search returned disjoint latent
sets. Two very different explanations for that:
  (a) real: the two methods target different things -- explanation search finds
      latents for *prose about* AI/alignment, the probe finds latents for text
      *doing* metagaming.
  (b) artifact: the probe's cross-source / doc-count filters simply excluded them.

This scores the 90 explanation-classified YES latents on the probe dataset with no
filters at all, so (a) and (b) can be told apart.
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/a5k/cwtice.a5k/sae-exploration/scripts")
from sae_local import Runner  # noqa: E402

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
SAE = "gemma3-l40-16k"

classified = [json.loads(l) for l in open(f"{ROOT}/out/classified.jsonl")]
yes = [r["index"] for r in classified if r["verdict"] == "YES"]
rows = [json.loads(l) for l in open(f"{ROOT}/data/probe_dataset.jsonl")]

runner = Runner(SAE)
tok = runner.tokenizer
bos = tok.bos_token_id


@torch.no_grad()
def acts_for(row):
    ids = ([bos] if bos is not None else []) + tok(
        row["text"], add_special_tokens=False, truncation=True,
        max_length=512)["input_ids"]
    a = runner.sae.encode(runner.residual(torch.tensor(ids)[None]))[0]
    return a.float().cpu().numpy()


pos_means, ctl_means = [], []
for r in rows:
    a = acts_for(r)
    (pos_means if r["label"] == "positive" else ctl_means).append(a.mean(0))
P = np.stack(pos_means).mean(0)
C = np.stack(ctl_means).mean(0)
fires_pos = (np.stack(pos_means) > 0).sum(0)

print(f"\n{len(yes)} explanation-classified YES latents, scored on the probe set\n")
print(f"{'latent':>7} {'mean_pos':>9} {'mean_ctl':>9} {'pos docs':>9}  description")
order = sorted(yes, key=lambda l: -P[l])
for lat in order[:15]:
    d = next(r["description"] for r in classified if r["index"] == lat)
    print(f"{lat:>7} {P[lat]:>9.2f} {C[lat]:>9.2f} {fires_pos[lat]:>6}/40  {d[:52]}")

silent = [l for l in yes if fires_pos[l] == 0]
weak = [l for l in yes if 0 < fires_pos[l] <= 4]
higher_on_ctl = [l for l in yes if P[l] < C[l]]
print(f"\nof the {len(yes)}: {len(silent)} never fire on any metagaming text, "
      f"{len(weak)} fire on <=4 of 40 docs, "
      f"{len(higher_on_ctl)} activate MORE on controls than on metagaming text")

probe = json.load(open(f"{ROOT}/out/probe_{SAE}.json"))
probe_lats = [r["latent"] for r in probe["ranked_by_contrast"]]
print(f"\nfor comparison, probe-discovered latents on the same measure:")
print(f"{'latent':>7} {'mean_pos':>9} {'mean_ctl':>9} {'pos docs':>9}")
for lat in probe_lats[:8]:
    print(f"{lat:>7} {P[lat]:>9.2f} {C[lat]:>9.2f} {fires_pos[lat]:>6}/40")
print(f"\nmedian mean_pos: explanation-set {np.median([P[l] for l in yes]):.2f} "
      f"vs probe-set {np.median([P[l] for l in probe_lats]):.2f}")
