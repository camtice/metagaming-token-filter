"""Token-level zoom for a chosen set of SAE latents over the probe docs.

Implements the measurement side of Rathi & Radford (arXiv 2601.21571) §5.1:
their seed rule labels a token as forget-domain when >= `min_features` of the
selected latents are >= 4 SD above their mean activation, then iteratively
grows spans onto adjacent tokens that have positive activation on >= 1 latent.
This export ships the RAW per-token activations plus per-latent stats so the
artifact can apply that rule interactively (SD multiplier, min-latents,
windowing on/off) rather than baking in one setting.

Emitted per document (NO text -- only [start,end) char offsets into the `text`
field of data/probe_dataset.jsonl, which the artifact already embeds):
  toks:   [[s,e], ...]      scored body tokens, in order
  acts:   {latent: [[tok_idx, act], ...]}   sparse, nonzero only
Per latent: mean/SD over all body tokens of the probe set (zeros included),
firing rate on the probe set, and the same stats restricted to control docs.

CAVEAT (HANDOFF open item 5): Rathi's 4-SD thresholds should be computed on
the corpus being filtered. The probe set is concept-dense, so mean/SD here
are inflated; stats over the 16 control docs are also exported as a colder
reference. Real filtering thresholds need a pretraining-corpus pass.

Usage:
  python export_token_view.py gemma3-l40-16k --latents 11620,11037,8552
  python export_token_view.py gemma3-l40-16k --latents-file latents.txt
Output: out/token_view_<sae>.json
"""
import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/a5k/cwtice.a5k/sae-exploration/scripts")
from sae_local import REGISTRY, Runner  # noqa: E402

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"

ap = argparse.ArgumentParser()
ap.add_argument("sae")
ap.add_argument("--latents", type=str, default=None, help="comma-separated latent ids")
ap.add_argument("--latents-file", type=str, default=None, help="file with one latent id per line")
ap.add_argument("--max-len", type=int, default=512)
args = ap.parse_args()

if args.latents:
    latents = sorted({int(x) for x in args.latents.split(",") if x.strip()})
elif args.latents_file:
    latents = sorted({int(l.split()[0]) for l in open(args.latents_file)
                      if l.strip() and not l.startswith("#")})
else:
    ap.error("give --latents or --latents-file")

cfg = REGISTRY[args.sae]
rows = [json.loads(l) for l in open(f"{ROOT}/data/probe_dataset.jsonl")]
runner = Runner(args.sae)
tok = runner.tokenizer
bos = tok.bos_token_id
lat_idx = torch.tensor(latents)


@torch.no_grad()
def encode(row):
    ctx = []
    if row.get("context"):
        ctx = tok(row["context"], add_special_tokens=False, truncation=True,
                  max_length=args.max_len // 2)["input_ids"]
    enc = tok(row["text"], add_special_tokens=False, truncation=True,
              max_length=args.max_len, return_offsets_mapping=True)
    body, offs = enc["input_ids"], enc["offset_mapping"]
    ids = ([bos] if bos is not None else []) + ctx + body
    ids = ids[-args.max_len:]
    n = min(len(body), len(ids))
    acts = runner.sae.encode(runner.residual(torch.tensor(ids)[None]))[0]
    # only the selected latents, body tokens only
    return acts[-n:, lat_idx].float().cpu().numpy(), offs[-n:]


docs = []
all_acts = []          # per doc: [n_tok, n_latents]
for row in rows:
    a, offs = encode(row)
    all_acts.append(a)
    sparse = {}
    for j, lat in enumerate(latents):
        nz = np.flatnonzero(a[:, j] > 0)
        if len(nz):
            sparse[str(lat)] = [[int(t), round(float(a[t, j]), 1)] for t in nz]
    docs.append({"id": row["id"], "toks": [[int(s), int(e)] for s, e in offs],
                 "acts": sparse})
    print(".", end="", flush=True)
print()

cat = np.concatenate(all_acts)                       # all body tokens, probe set
ctl = np.concatenate([a for a, r in zip(all_acts, rows) if r["label"] == "control"])
stats = {}
for j, lat in enumerate(latents):
    v, c = cat[:, j], ctl[:, j]
    stats[str(lat)] = {
        "mean": round(float(v.mean()), 4), "sd": round(float(v.std()), 4),
        "fire_rate": round(float((v > 0).mean()), 5),
        "ctl_mean": round(float(c.mean()), 4), "ctl_sd": round(float(c.std()), 4),
        "ctl_fire_rate": round(float((c > 0).mean()), 5),
        "max": round(float(v.max()), 1),
    }

out = {
    "sae": args.sae, "latents": latents, "n_body_tokens": int(cat.shape[0]),
    "n_ctl_tokens": int(ctl.shape[0]),
    "stats_note": ("mean/sd computed over the probe set's body tokens (zeros "
                   "included) and, as a colder reference, over control docs only. "
                   "The probe set is concept-dense -- real Rathi thresholds need "
                   "pretraining-corpus statistics (HANDOFF open item 5)."),
    "rule_note": ("Rathi & Radford 2601.21571 SS5.1: seed = act >= mean + 4*SD on "
                  ">=2 selected latents; grow = positive act on >=1 latent while "
                  "adjacent to a labeled token, iterate to convergence."),
    "stats": stats,
    "docs": docs,
}
p = f"{ROOT}/out/token_view_{args.sae}.json"
json.dump(out, open(p, "w"), separators=(",", ":"))
print(f"wrote {p}")
nz = sum(len(v) for d in docs for v in d["acts"].values())
print(f"  latents: {len(latents)}, nonzero (tok,latent) acts: {nz}, "
      f"body tokens: {cat.shape[0]}")
