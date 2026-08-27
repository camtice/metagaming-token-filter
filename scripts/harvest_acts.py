"""Harvest per-token SAE activations for a document set, once, to disk.

One GPU pass -> sparse (doc, token, latent, act) store + token char offsets.
Every downstream stage (FP audit, expansion, selection, validation) then runs
CPU-only from this file, which makes the analysis exactly reproducible and the
frozen validation a pure re-run on a new harvest.

Long documents are chunked into BOS + 1023-token windows with 64-token overlap;
overlapping tokens are scored once (first occurrence). Docs may carry an
optional `context` field (prepended, never scored), like the probe set.

Usage: python harvest_acts.py gemma3-l40-16k data/dev_docs_pr123.jsonl out/harvest_dev
Writes <out>.npz (doc_idx/tok_idx/lat_idx/act int32+fp16 arrays, offsets, doc token counts)
   and <out>.meta.json (doc metadata in order, no text).
"""
import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/a5k/cwtice.a5k/sae-exploration/scripts")
from sae_local import REGISTRY, Runner  # noqa: E402

WIN, OVERLAP = 1023, 64

ap = argparse.ArgumentParser()
ap.add_argument("sae")
ap.add_argument("docs_jsonl")
ap.add_argument("out_prefix")
args = ap.parse_args()

cfg = REGISTRY[args.sae]
rows = [json.loads(l) for l in open(args.docs_jsonl)]
runner = Runner(args.sae)
tok, bos = runner.tokenizer, runner.tokenizer.bos_token_id

d_doc, d_tok, d_lat, d_act = [], [], [], []
offsets, doc_ntok = [], []

@torch.no_grad()
def doc_acts(row):
    """Sparse acts + char offsets for every scored body token of one doc."""
    ctx = []
    if row.get("context"):
        ctx = tok(row["context"], add_special_tokens=False, truncation=True,
                  max_length=WIN // 2)["input_ids"]
    enc = tok(row["text"], add_special_tokens=False, return_offsets_mapping=True)
    body, offs = enc["input_ids"], enc["offset_mapping"]
    pieces_a, pieces_o = [], []
    start = 0
    first = True
    while start < len(body):
        left = ctx if first else body[max(0, start - OVERLAP):start]
        left = left[-(WIN - 1):]
        seg = body[start:start + (WIN - len(left))]
        ids = ([bos] if bos is not None else []) + left + seg
        acts = runner.sae.encode(runner.residual(torch.tensor(ids)[None]))[0]
        pieces_a.append(acts[-len(seg):].float().cpu())
        pieces_o.append(offs[start:start + len(seg)])
        start += len(seg)
        first = False
    return torch.cat(pieces_a), [o for p in pieces_o for o in p]

for di, row in enumerate(rows):
    acts, offs = doc_acts(row)
    nz = (acts > 0).nonzero()
    d_doc.append(np.full(len(nz), di, dtype=np.int32))
    d_tok.append(nz[:, 0].numpy().astype(np.int32))
    d_lat.append(nz[:, 1].numpy().astype(np.int32))
    d_act.append(np.clip(acts[nz[:, 0], nz[:, 1]].numpy(), 0, 65504).astype(np.float16))
    offsets.append(np.array(offs, dtype=np.int32))
    doc_ntok.append(len(offs))
    print(".", end="", flush=True)
print()

np.savez_compressed(
    args.out_prefix + ".npz",
    doc_idx=np.concatenate(d_doc), tok_idx=np.concatenate(d_tok),
    lat_idx=np.concatenate(d_lat), act=np.concatenate(d_act),
    doc_ntok=np.array(doc_ntok, dtype=np.int32),
    offsets=np.concatenate(offsets), offsets_doc_len=np.array([len(o) for o in offsets]))
meta = [{k: r.get(k) for k in ("id", "group", "origin", "topic", "axis", "category", "label", "source")}
        for r in rows]
json.dump({"sae": args.sae, "width": cfg["width"], "docs": meta,
           "window": WIN, "overlap": OVERLAP},
          open(args.out_prefix + ".meta.json", "w"), indent=1)
nnz = sum(len(x) for x in d_act)
print(f"wrote {args.out_prefix}.npz | docs={len(rows)} tokens={sum(doc_ntok)} nnz={nnz} "
      f"(density {nnz / max(sum(doc_ntok), 1):.1f} lat/tok)")
