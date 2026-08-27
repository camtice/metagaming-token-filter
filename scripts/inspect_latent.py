"""Score named latents on the probe set, whether or not they made a top-N list.

Answers "does latent X fire on our metagaming data, and on which axis?" for any
latent, including ones the rankings dropped. Prints per-axis means, the AUC it
would have scored, and its strongest passage in each axis.

Usage: python inspect_latent.py gemma3-l40-16k 4936 [1318 ...]
"""
import argparse
import glob
import gzip
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/a5k/cwtice.a5k/sae-exploration/scripts")
from sae_local import REGISTRY, Runner, sae_dir  # noqa: E402

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
ap = argparse.ArgumentParser()
ap.add_argument("sae")
ap.add_argument("latents", type=int, nargs="+")
ap.add_argument("--max-len", type=int, default=512)
args = ap.parse_args()

cfg = REGISTRY[args.sae]
rows = [json.loads(l) for l in open(f"{ROOT}/data/probe_dataset.jsonl")]
runner = Runner(args.sae)
tok, bos = runner.tokenizer, runner.tokenizer.bos_token_id
np_model, np_source = cfg["neuronpedia"] or (None, None)

descs = {}
if np_source:
    for fn in glob.glob(f"{ROOT}/out/np*batch-*.jsonl.gz"):
        for line in gzip.open(fn, "rt"):
            d = json.loads(line)
            if d.get("layer") == np_source:
                descs[int(d["index"])] = d["description"]

AXES = ["deception_metagaming", "metagaming_clean", "deception_generic", "control_other"]
acc = {a: {l: [] for l in args.latents} for a in AXES}
best = {l: {} for l in args.latents}


@torch.no_grad()
def run(row):
    ctx = []
    if row.get("context"):
        ctx = tok(row["context"], add_special_tokens=False, truncation=True,
                  max_length=args.max_len // 2)["input_ids"]
    body = tok(row["text"], add_special_tokens=False, truncation=True,
               max_length=args.max_len)["input_ids"]
    ids = (([bos] if bos is not None else []) + ctx + body)[-args.max_len:]
    n = min(len(body), len(ids))
    a = runner.sae.encode(runner.residual(torch.tensor(ids)[None]))[0]
    return a[-n:].float().cpu().numpy(), ids[-n:]


for row in rows:
    acts, ids = run(row)
    for l in args.latents:
        v = acts[:, l]
        acc[row["axis"]][l].append(float(v.mean()))
        if v.max() > best[l].get(row["axis"], (0,))[0]:
            p = int(v.argmax())
            lo, hi = max(0, p - 14), min(len(ids), p + 15)
            best[l][row["axis"]] = (
                float(v.max()), row["id"],
                tok.decode(ids[lo:p]) + "«" + tok.decode([ids[p]]) + "»" + tok.decode(ids[p + 1:hi]))
    print(".", end="", flush=True)
print("\n")

for l in args.latents:
    m = {a: (np.mean(acc[a][l]) if acc[a][l] else 0.0) for a in AXES}
    pos = np.array(acc["deception_metagaming"][l] + acc["metagaming_clean"][l])
    ctl = np.array(acc["deception_generic"][l] + acc["control_other"][l])
    wins = (pos[:, None] > ctl[None, :]).sum() + 0.5 * (pos[:, None] == ctl[None, :]).sum()
    auc = wins / (len(pos) * len(ctl))
    nz = {a: sum(1 for x in acc[a][l] if x > 0) for a in AXES}

    print("=" * 100)
    print(f"LATENT {l}   {descs.get(l, '(no published label)')!r}")
    if np_model:
        print(f"  https://www.neuronpedia.org/{np_model}/{np_source}/{l}")
    print(f"  AUC (metagaming vs controls) = {auc:.3f}")
    print(f"\n  {'axis':<22}{'mean act':>10}{'docs firing':>14}")
    for a in AXES:
        print(f"  {a:<22}{m[a]:>10.1f}{nz[a]:>8}/{len(acc[a][l]):<5}")
    print()
    for a in AXES:
        if a in best[l]:
            v, doc, txt = best[l][a]
            print(f"  [{a}] peak {v:.0f} in {doc}")
            print(f"     {txt}")
    print()
